# 0. Prepare custom libs
1. Go to: https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes/actions/workflows/build-rccl.yml
2. Click on the latest successful run.
3. Scroll down to "Artifacts" and download `librccl-gfx1151`.
4. Extract the zip file to get `librccl.so.1.gz`.
5. Create a folder `custom_libs` in this directory: `mkdir -p custom_libs`
6. Move the file there: `mv /path/to/downloaded/librccl.so.1.gz custom_libs/`

# 1. Build container
`podman build -t llm-finetuning .`

# 2. Log in to Docker Hub
`podman login docker.io`

# 3. Tag your local image for Docker Hub
`podman tag llm-finetuning docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest`

# 4. Push it
`podman push docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest`
