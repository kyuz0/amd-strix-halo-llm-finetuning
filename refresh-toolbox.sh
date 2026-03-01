#!/usr/bin/env bash
set -euo pipefail

NAME="strix-halo-llm-finetuning"
IMAGE="docker.io/kyuz0/amd-strix-halo-llm-finetuning:latest"
REPO="${IMAGE%:*}"  # docker.io/kyuz0/amd-strix-halo-llm-finetuning

echo "Checking remote digest for $IMAGE ..."
remote_digest="$(skopeo inspect docker://$IMAGE | jq -r '.Digest' 2>/dev/null || true)"

if [[ -z "$remote_digest" || "$remote_digest" == "null" ]]; then
  echo "Could not resolve remote digest for $IMAGE"
  exit 1
fi

# Check if the currently tagged image matches the remote digest
local_repo_digests="$(podman image inspect --format '{{.RepoDigests}}' "$IMAGE" 2>/dev/null || true)"
if [[ "$local_repo_digests" == *"$remote_digest"* ]]; then
  echo "Already up to date."
  exit 0
fi

# Store the old image ID before we pull the new one
old_image_id="$(podman image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"

echo "Updating $IMAGE ..."
podman pull "$IMAGE"

new_image_id="$(podman image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"

# Base options
OPTIONS="--device /dev/dri --device /dev/kfd --group-add video --group-add render --security-opt seccomp=unconfined"

# Check for InfiniBand devices (needed for RDMA/RoCE in multi-node training)
if [ -d "/dev/infiniband" ]; then
    echo "🔎 InfiniBand devices detected! Adding RDMA support..."
    OPTIONS="$OPTIONS --device /dev/infiniband --group-add rdma --ulimit memlock=-1"
else
    echo "ℹ️  No InfiniBand devices detected (RDMA will not be available)."
fi

echo "Recreating toolbox $NAME ..."
toolbox rm -f "$NAME" 2>/dev/null || true
toolbox create "$NAME" \
  --image "$IMAGE" \
  -- $OPTIONS

# Clean up only the old image that we just replaced
if [[ -n "$old_image_id" && "$old_image_id" != "$new_image_id" ]]; then
  echo "Removing previous image ($old_image_id) ..."
  podman image rm -f "$old_image_id" 2>/dev/null || true
fi

echo "Done."
