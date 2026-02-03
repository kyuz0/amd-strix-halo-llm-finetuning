#!/bin/bash
set -e

# Configuration
REPO="kyuz0/amd-strix-halo-vllm-toolboxes"
WORKFLOW="build-rccl.yml"
ARTIFACT_NAME="librccl-gfx1151"
IMAGE_NAME="docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest"

# Check dependencies
if ! command -v gh &> /dev/null; then
    echo "Error: 'gh' CLI not found. Please install manually."
    exit 1
fi

if ! command -v podman &> /dev/null; then
    echo "Error: 'podman' not found. Please install manually."
    exit 1
fi

echo "=== 1. Preparing Custom Libs ==="
mkdir -p custom_libs

echo "Fetching latest successful run for $WORKFLOW in $REPO..."
RUN_ID=$(gh run list --repo "$REPO" --workflow "$WORKFLOW" --status success --limit 1 --json databaseId --jq '.[0].databaseId')

if [ -z "$RUN_ID" ]; then
    echo "Error: No successful run found for $WORKFLOW."
    exit 1
fi

echo "Downloading artifact '$ARTIFACT_NAME' from run $RUN_ID..."
gh run download "$RUN_ID" --repo "$REPO" --name "$ARTIFACT_NAME" --dir custom_libs

echo "Extracting artifact..."
if [ -f "custom_libs/librccl.so.1.gz" ]; then
    echo "Artifact found: custom_libs/librccl.so.1.gz"
else
    echo "Error: expected 'custom_libs/librccl.so.1.gz' not found after download."
    ls -lh custom_libs/
    exit 1
fi

echo "=== 2. Building Container ==="
podman build -t llm-finetuning .

echo "=== 3. Login to Docker Hub ==="
podman login docker.io

echo "=== 4. Tagging Image ==="
podman tag llm-finetuning "$IMAGE_NAME"

echo "=== 5. Pushing Image ==="
podman push "$IMAGE_NAME"

echo "Done! Image pushed to $IMAGE_NAME"
