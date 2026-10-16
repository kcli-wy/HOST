#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ALIGNMENT_PYTHON=${ALIGNMENT_PYTHON:-/data/250010175/miniforge3/envs/HOST_Alignment/bin/python}
ALIGNMENT_ENV=$(cd -- "$(dirname -- "$ALIGNMENT_PYTHON")/.." && pwd)
export PATH="$ALIGNMENT_ENV/bin:$PATH"
if [ -x "$ALIGNMENT_ENV/bin/nvcc" ]; then
    export CUDA_HOME="$ALIGNMENT_ENV"
    export CUDA_PATH="$ALIGNMENT_ENV"
    export CUDAToolkit_ROOT="$ALIGNMENT_ENV"
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=${WANDB_MODE:-offline}
fi

exec "$ALIGNMENT_PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node="${NGPUS:-1}" \
    "$SCRIPT_DIR/train_humangen.py" "$@"
