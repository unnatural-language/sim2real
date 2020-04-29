#!/bin/bash

# fill this in
YOUR_MODEL=

python -u scripts/evaluate_turk.py \
  --env BabyAI-SynthLoc-v0 \
  --model $YOUR_MODEL \
  --turk_file test/annotations.csv \
  --human
