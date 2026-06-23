#!/usr/bin/env bash
# Phase-3 sweep launcher (W18-W23), in the locked order:
#   W18 PPO K=1 -> W19 PPO K=3 -> W20 TD3 K=1 -> W21 TD3 K=3 -> W22 SAC K=1 -> W23 SAC K=3
#
# Each run uses the --macro sweep config (UMa @ 1 km), auto-saves every episode,
# and supports --resume. Per the week-audit-gate protocol, STOP after each week
# and have the formulation audit Gate 3x pass before starting the next.
#
# Usage:
#   cd baselines
#   EPISODES=10000 SEED=0 bash scripts/run_sweep.sh w18      # one week
#   EPISODES=500   SEED=0 bash scripts/run_sweep.sh w18      # pilot
#   EPISODES=10000 SEEDS="0 1 2 3 4" bash scripts/run_sweep.sh w18   # multi-seed
#
# Env vars:
#   EPISODES  (default 10000)   number of episodes
#   SEED      (default 0)       single seed (ignored if SEEDS set)
#   SEEDS     (optional)        space-separated seeds -> runs each sequentially
#   RESUME    (default off)     set RESUME=1 to add --resume (continue an
#                               interrupted run). Leave UNSET for a fresh start —
#                               otherwise a stale checkpoint warm-starts the run.
#   EXTRA     (optional)        extra flags appended verbatim (e.g. "--wandb")
set -euo pipefail

EPISODES="${EPISODES:-10000}"
SEED="${SEED:-0}"
SEEDS="${SEEDS:-$SEED}"
EXTRA="${EXTRA:-}"
RESUME_FLAG=""; [ -n "${RESUME:-}" ] && RESUME_FLAG="--resume"
WEEK="${1:?usage: run_sweep.sh <w18|w19|w20|w21|w22|w23|all>}"

run_ppo()   { local k=$1 s=$2; python3 train.py --algo ppo --macro --K "$k" \
                --episodes "$EPISODES" --seed "$s" $RESUME_FLAG $EXTRA; }
run_offp()  { local algo=$1 k=$2 s=$3; python3 -m "solvers.train_$algo" \
                --macro --K "$k" --episodes "$EPISODES" --seed "$s" $RESUME_FLAG $EXTRA; }

for s in $SEEDS; do
  echo "=================  $WEEK  seed=$s  episodes=$EPISODES  ================="
  case "$WEEK" in
    w18) run_ppo  1 "$s" ;;
    w19) run_ppo  3 "$s" ;;
    w20) run_offp td3 1 "$s" ;;
    w21) run_offp td3 3 "$s" ;;
    w22) run_offp sac 1 "$s" ;;
    w23) run_offp sac 3 "$s" ;;
    all) run_ppo 1 "$s"; run_ppo 3 "$s";
         run_offp td3 1 "$s"; run_offp td3 3 "$s";
         run_offp sac 1 "$s"; run_offp sac 3 "$s" ;;
    *) echo "unknown week: $WEEK" >&2; exit 2 ;;
  esac
done
echo "DONE $WEEK. Summaries: logs/summary_<algo>_seed<seed>.json | checkpoints/"
echo "NEXT: run the Gate-3x formulation audit before the next week."
