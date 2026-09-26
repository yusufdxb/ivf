#!/bin/bash
cd ~/Projects/isaac-sim-contrib/IsaacLab
for spec in "Isaac-Velocity-Flat-UnitreeGo2 300" "Isaac-Velocity-Flat-AnymalD 300" "Isaac-Velocity-Flat-H1 600" "Isaac-Velocity-Flat-G1 800"; do
  set -- $spec
  echo "=== START $1 iters=$2 $(date -Is)"
  env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 ./env_isaaclab/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
     --task $1 --headless --seed 42 --max_iterations $2 > /tmp/claude-1000/-home-yusuf/1e2fa97f-81ba-4e8b-bd38-c8495af4a800/scratchpad/train/$1.log 2>&1
  echo "=== END $1 exit=$? $(date -Is)"
done
