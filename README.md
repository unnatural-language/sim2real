# sim2real
Evaluation data and code for BabyAI experiments in "Unnatural Language
Processing: Bridging the Gap Between Synthetic and Natural Language Data" (https://arxiv.org/abs/2004.13645) by Marzoev et al.

This code is meant to be used with the BabyAI package by Boisvert et al., with
code available at https://github.com/mila-iqia/babyai.

For our evaluations,
- The original `babyai/evaluate.py` should be replaced with the corresponding
  file provided here.
- `evaluate_turk.py` should be placed in the `scripts` directory.
- `run_eval.sh` should be run from the project root.

The data release includes fine-tuning and test data. Each directory contains
both a set of Mechanical Turk annotations and a set of demonstrated
trajectories. Correspondence between trajectories and annotations can be
determiend based on the seed; the trajectories are for seeds 1--5000 in order
for fine_tuning data, and 1000000--1000509 for test data. Seeds are also used to
initialize the BabyAI simulator (re-seeding is the only way to bring BabyAI into
a known state); see evaluate.py for details.
 
