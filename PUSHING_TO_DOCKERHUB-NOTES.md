# 0. Build container
`podman build -t llm-finetuning .`

# 1. Log in to Docker Hub
`podman login docker.io`

# 2. Tag your local image for Docker Hub
`podman tag llm-finetuning docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest`

# 3. Push it
`podman push docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest`
