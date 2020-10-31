# Copyright (c) Facebook, Inc. and its affiliates.

# Debugging cycle consistency transfer.

python Master.py --name=CTdebug --train=1 --setting=cycle_transfer --source_domain=ContinuousNonZero --target_domain=ContinuousNonZero --z_dimensions=64 --number_layers=5 --hidden_size=64 --data=ContinuousNonZero --training_phase_size=10000 --display_freq=1000 --eval_freq=4 --alternating_phase_size=200 --discriminator_phase_size=2 --vae_loss_weight=1. --discriminability_weight=2.0 --kl_weight=0.001
