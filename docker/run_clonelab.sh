#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

docker compose \
    --file docker-compose.yaml \
    --file docker-compose.clonelab.yaml \
    up rover-lab-base --detach --build --remove-orphans
