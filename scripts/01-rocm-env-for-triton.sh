#!/usr/bin/env bash
# Detect and export ROCm toolchain paths from the _rocm_sdk_core package

# Query Python for the embedded ROCm paths
eval "$(
python3 - <<'PY'
import pathlib, _rocm_sdk_core as r
base = pathlib.Path(r.__file__).parent / "lib" / "llvm" / "bin"
lib  = pathlib.Path(r.__file__).parent / "lib"
print(f'export TRITON_HIP_LLD_PATH="{base / "ld.lld"}"')
print(f'export TRITON_HIP_CLANG_PATH="{base / "clang++"}"')
print(f'export LD_LIBRARY_PATH="{lib}:$LD_LIBRARY_PATH"')
PY
)"

# Enable Triton AMD backend for flash-attn
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
