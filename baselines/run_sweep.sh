#!/usr/bin/env bash
# Sweep driver: PPO only x 5 seeds x {K=1:1500ep, K=3:2000ep} = 10 runs.
# (TD3/SAC dropped 2026-06-24 per user request.)
# GPU, parallel (concurrency cap), per-episode autosave + --resume (idempotent:
# re-run this script to CONTINUE any unfinished run from its latest checkpoint).
#
#   bash run_sweep.sh            # launch / resume all 10
#   tail -f logs/sweep/_driver.log
set -u
cd "$(dirname "$0")"            # = baselines/
mkdir -p logs/sweep

# One thread per process — 10 procs on 12 cores; prevents BLAS/torch oversubscription.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

P=10                            # max concurrent runs (12 cores, leave 2)
CKPT_EVERY=100
LOGDIR=logs/sweep

DRIVER_LOG="$LOGDIR/_driver.log"
echo "=== sweep start $(date -Is) | P=$P ===" >> "$DRIVER_LOG"

# Job list: "algo K seed episodes". PPO only. K=3 (longer) first.
JOBS=()
for seed in 0 1 2 3 4; do JOBS+=("ppo 3 $seed 2000"); done
for seed in 0 1 2 3 4; do JOBS+=("ppo 1 $seed 1500"); done

launch() {
  local algo=$1 K=$2 seed=$3 ep=$4
  local name="${algo}_K${K}_seed${seed}"
  local state="$LOGDIR/${name}/checkpoints/${algo}_seed${seed}_state.json"
  # Skip if already at/past target episodes (idempotent re-run).
  if [ -f "$state" ]; then
    local last
    last=$(python3 -c "import json,sys; print(json.load(open('$state')).get('last_ep',0))" 2>/dev/null || echo 0)
    if [ "${last:-0}" -ge "$ep" ]; then
      echo "[skip] $name already at ep $last >= $ep" >> "$DRIVER_LOG"; return 0
    fi
  fi
  echo "[start $(date +%H:%M:%S)] $name (target $ep ep)" >> "$DRIVER_LOG"
  python3 train.py --algo "$algo" --macro --K "$K" --seed "$seed" --episodes "$ep" \
      --device cuda --log-dir "$LOGDIR" --checkpoint-every "$CKPT_EVERY" \
      --resume --print-every 25 > "$LOGDIR/${name}.out" 2>&1
  echo "[done  $(date +%H:%M:%S)] $name exit=$?" >> "$DRIVER_LOG"
}

running=0
for job in "${JOBS[@]}"; do
  # shellcheck disable=SC2086
  set -- $job
  launch "$1" "$2" "$3" "$4" &
  running=$((running+1))
  sleep 3                        # stagger CUDA init to smooth the initial burst
  if [ "$running" -ge "$P" ]; then wait -n; running=$((running-1)); fi
done
wait
echo "=== sweep DONE $(date -Is) ===" >> "$DRIVER_LOG"
