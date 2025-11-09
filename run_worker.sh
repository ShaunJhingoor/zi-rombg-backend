#!/bin/bash
set -e

CONTAINER_NAME="ai-bg-worker-container"
IMAGE_NAME="ai-bg-worker"
PORT="8080"

echo "🧹 Stopping any running container named $CONTAINER_NAME..."
docker stop $CONTAINER_NAME >/dev/null 2>&1 || true

echo "🗑️  Removing old container..."
docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true

echo "⚙️  Building image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

echo "🚀 Running container (visible logs, auto-clean on stop)..."
docker run --rm --name $CONTAINER_NAME -p $PORT:8080 $IMAGE_NAME

# --- optional cleanups after you stop it ---
echo "🧹 Pruning stopped containers and dangling images..."
docker container prune -f >/dev/null
docker image prune -f >/dev/null

# ./run_worker.sh