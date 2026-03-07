#!/bin/bash
set -e

# Configuration
IMAGE_NAME="docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest"

# Check dependencies
if ! command -v podman &> /dev/null; then
    echo "Error: 'podman' not found. Please install manually."
    exit 1
fi

echo "=== 1. Building Container ==="
podman build --no-cache -t llm-finetuning .

echo "=== 2. Login to Docker Hub ==="
podman login docker.io

echo "=== 3. Tagging Image ==="
podman tag llm-finetuning "$IMAGE_NAME"

echo "=== 4. Pushing Image ==="
podman push "$IMAGE_NAME"

echo "Done! Image pushed to $IMAGE_NAME"
