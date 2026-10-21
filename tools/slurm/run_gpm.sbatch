#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

# Optional template. Supply your own resources to sbatch, for example:
# sbatch --time=01:00:00 --cpus-per-task=4 --mem=16G --export=ALL tools/slurm/run_gpm.sbatch
# This job connects to already-running VLM services; it does not allocate or start them.
# Export GPM_REPO as the absolute checkout path before submission.
set -euo pipefail
: "${GPM_REPO:?Set GPM_REPO to the absolute release checkout path}"
GPM_CONFIG=${GPM_CONFIG:-configs/example.yaml}
GPM_PARADIGM=${GPM_PARADIGM:-windowed}
GPM_LABEL=${GPM_LABEL:-slurm-${SLURM_JOB_ID:-manual}}
PYTHON=${PYTHON:-python}
case "$GPM_PARADIGM" in
  windowed|streaming) ;;
  *) printf 'GPM_PARADIGM must be windowed or streaming\n' >&2; exit 2 ;;
esac
cd -- "$GPM_REPO"
exec "$PYTHON" -m memoproagent.cli.run --config "$GPM_CONFIG" \
  --paradigm "$GPM_PARADIGM" --label "$GPM_LABEL" --no-judge
