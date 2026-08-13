#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export HF_CACHE_HOST_PATH="${HF_CACHE_HOST_PATH:-${HOME}/.cache/huggingface}"
if [[ ! -d "${HF_CACHE_HOST_PATH}" ]]; then
    echo "Hugging Face cache does not exist: ${HF_CACHE_HOST_PATH}" >&2
    exit 1
fi

docker compose \
    --file docker-compose.yaml \
    build rover-lab-base

docker build \
    --file Dockerfile.rc-eval \
    --tag rover-lab-rc-eval:latest \
    .

docker compose \
    --file docker-compose.yaml \
    --file docker-compose.clonelab.yaml \
    --file docker-compose.rc-eval.yaml \
    up rover-lab-base --detach --force-recreate --remove-orphans
