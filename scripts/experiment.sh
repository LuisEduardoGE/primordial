#!/usr/bin/env bash
# Old-style (fixed, uniform) vs new-style (self-adaptive, per-layer, neuron resets) mutation.
# Same world (memory + wandering food patches) for both; only mutation differs.
set -e
mkdir -p "$(dirname "$0")/../runs"
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
OLD="sigma_tau=0 neuron_reset_prob=0 perception_rate_mult=1 behaviour_rate_mult=1 attraction_rate_mult=1"
export OMP_NUM_THREADS=3
run() {  # name seed extra-sets...
  local name=$1 seed=$2; shift 2
  $PY -m primordial run --seed $seed --steps 5000 --every 1000 --log runs/$name.5k.csv --save runs/$name.5k.pt --set "$@" > runs/$name.log
  $PY -m primordial run --load runs/$name.5k.pt --steps 10000 --every 1000 --log runs/$name.15k.csv --save runs/$name.15k.pt >> runs/$name.log
}
run old_s1 1 $OLD &
run old_s2 2 $OLD &
run new_s1 1 seed=1 &
run new_s2 2 seed=2 &
wait
echo EXPERIMENT_DONE
