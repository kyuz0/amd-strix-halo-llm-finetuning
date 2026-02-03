FROM registry.fedoraproject.org/fedora:43 AS builder

# --- base toolchain ---
# matched with tmp-vllm/scripts/install_deps.sh
RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
    make gcc gcc-c++ cmake lld clang clang-devel compiler-rt libcurl-devel \
    radeontop git vim patch curl ninja-build tar libatomic xz \
    python3.13 python3.13-devel pip aria2c jupyterlab \
    gperftools-libs libdrm-devel zlib-devel openssl-devel numactl-devel \
    libibverbs-utils perftest \
    && dnf clean all && rm -rf /var/cache/dnf/*

# --- fetch and unpack ROCm TheRock ---
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
RUN set -euo pipefail; \
    BASE="https://therock-nightly-tarball.s3.amazonaws.com"; \
    PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"; \
    KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
    | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\.[0-9]\+\.[0-9]\+rc[0-9]\{8\}\.tar\.gz" \
    | sort | tail -n1)"; \
    echo "Latest tarball: ${KEY}"; \
    aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz
RUN mkdir -p /opt/rocm-7.0 && \
    tar xzf therock.tar.gz -C /opt/rocm-7.0 --strip-components=1

# --- ROCm env vars ---
ENV ROCM_PATH=/opt/rocm-7.0 \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm-7.0 \
    HIP_CLANG_PATH=/opt/rocm-7.0/llvm/bin \
    HIP_INCLUDE_PATH=/opt/rocm-7.0/include \
    HIP_LIB_PATH=/opt/rocm-7.0/lib \
    HIP_DEVICE_LIB_PATH=/opt/rocm-7.0/lib/llvm/amdgcn/bitcode \
    LD_LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64:/opt/rocm-7.0/llvm/lib \
    LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64 \
    CPATH=/opt/rocm-7.0/include \
    PKG_CONFIG_PATH=/opt/rocm-7.0/lib/pkgconfig \
    PYTHONNOUSERSITE=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4

RUN printf '%s\n' \
    'export ROCM_PATH=/opt/rocm-7.0' \
    'export HIP_PLATFORM=amd' \
    'export HIP_PATH=/opt/rocm-7.0' \
    'export HIP_CLANG_PATH=/opt/rocm-7.0/llvm/bin' \
    'export HIP_INCLUDE_PATH=/opt/rocm-7.0/include' \
    'export HIP_LIB_PATH=/opt/rocm-7.0/lib' \
    'export HIP_DEVICE_LIB_PATH=/opt/rocm-7.0/lib/llvm/amdgcn/bitcode' \
    'export LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib"' \
    'export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"' \
    'export CPATH="$HIP_INCLUDE_PATH"' \
    'export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"' \
    'export ROCBLAS_USE_HIPBLASLT=1' \
    'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1' \
    'export PYTHONNOUSERSITE=1' \
    'export LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4' \
    > /etc/profile.d/rocm.sh && chmod +x /etc/profile.d/rocm.sh && echo 'source /etc/profile.d/rocm.sh' >> /etc/bashrc

# --- create venv to keep one consistent Python interpreter ---
RUN /usr/bin/python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python -m pip install --no-cache-dir -U pip setuptools wheel packaging

# --- ROCm PyTorch ---
# Update to v2-staging as per tmp-vllm
RUN python -m pip install --no-cache-dir \
    --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ \
    --pre torch torchvision

# --- bitsandbytes (ROCm) ---
WORKDIR /opt
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git
WORKDIR /opt/bitsandbytes
RUN cmake -S . -DGPU_TARGETS="gfx1151" -DBNB_ROCM_ARCH="gfx1151" -DCOMPUTE_BACKEND=hip && \
    make -j && \
    python -m pip install --no-cache-dir . --no-build-isolation --no-deps

# --- Flash-Attention ---
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
WORKDIR /opt
RUN git clone https://github.com/ROCm/flash-attention.git && \
    cd flash-attention && \
    python -m pip install --no-cache-dir packaging && \
    git checkout main_perf && \
    python setup.py install && \
    cd /opt && rm -rf /opt/flash-attention

# --- LLM runtime stack ---
# Hugging Face stack (same interpreter, safe with ROCm PyTorch)
RUN python -m pip install --no-cache-dir \
    jq \
    transformers \
    peft \
    accelerate \
    safetensors \
    sentencepiece \
    huggingface-hub \
    trl \
    einops \
    tqdm==4.67.1 \
    ipywidgets==8.1.7 \
    ipykernel==6.30.1 \
    traitlets==5.14.3 \
    jupyter_core==5.8.1

# Copy workspace
COPY workspace /opt/workspace

# --- cleanup ---
RUN chmod -R a+rwX /opt && \
    python -m pip cache purge || true && rm -rf /root/.cache/pip || true && \
    dnf clean all && rm -rf /var/cache/dnf/*

# --- env scripts ---
COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
RUN chmod 0644 /etc/profile.d/*.sh

# --- disable core dumps ---
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

# Force Jupyter to default to the venv interpreter
RUN /opt/venv/bin/python -m ipykernel install --name=venv --display-name "Python (venv)" --prefix=/usr/local && \
    mkdir -p /root/.local/share/jupyter/kernels/python3 && \
    jq '.argv[0] = "/opt/venv/bin/python"' /usr/local/share/jupyter/kernels/venv/kernel.json > /root/.local/share/jupyter/kernels/python3/kernel.json


CMD ["/bin/bash"]
