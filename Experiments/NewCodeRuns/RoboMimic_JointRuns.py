# Template MIME run
python Master.py --train=1 --setting=learntsub --name=MJ_rerun_116 --normalization=minmax --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=MIME  --subpolicy_model=ExpWandbLogs/MBP_111/saved_models/Model_epoch485 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=1 --short_trajectories=1 --epsilon_from=0.4 --epsilon_to=0.01 --epsilon_over=100

# Now running...
# using RMP_005 -- minmax, and RMP_007 -- meanvar 
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=RM_001 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_005/saved_models/Model_epoch2000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000 --normalization=minmax

# 
python Master.py --train=1 --setting=learntsub --name=RM_002 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_007/saved_models/Model_epoch2000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000 --normalization=meanvar

# No norm
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=RM_003 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_004/saved_models/Model_epoch500 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000

# Debug segfault
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=RM_debug_segfault --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_005/saved_models/Model_epoch2000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000 --normalization=minmax

# Continue
CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=learntsub --name=RM_001_cont --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --model=ExpWandbLogs/RM_001/saved_models/Model_epoch1000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000 --normalization=minmax

# 
python Master.py --train=1 --setting=learntsub --name=RM_002_cont1 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --model=ExpWandbLogs/RM_002/saved_models/Model_epoch1000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000 --normalization=meanvar

# Rerun with the visualization
python Master.py --train=1 --setting=learntsub --name=RM_testviz --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --model=ExpWandbLogs/RM_002/saved_models/Model_epoch1000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=2000 --normalization=meanvar

# Run joint RM training with 010 and 011 models.
# RMP_010 - minmax, RMP_011 - meanvar
#
CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=learntsub --name=RM_003 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_010/saved_models/Model_epoch5000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=5000 --normalization=minmax --save_freq=100

# 
CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=learntsub --name=RM_004 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_011/saved_models/Model_epoch5000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=5000 --normalization=meanvar --save_freq=100

# Also running on hinton for speed check
#
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=learntsub --name=RM_005 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_010/saved_models/Model_epoch5000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=5000 --normalization=minmax --save_freq=100

# 
CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=learntsub --name=RM_006 --normalization=meanvar --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_011/saved_models/Model_epoch5000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=5000 --normalization=meanvar --save_freq=100

# Running joint training with new skill segmentation lengths
#
CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=learntsub --name=RM_007 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_010/saved_models/Model_epoch5000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=5000 --normalization=minmax --save_freq=500

# 
CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=learntsub --name=RM_008 --kl_weight=0.001 --subpolicy_ratio=1. --latentpolicy_ratio=0.0 --b_probability_factor=0.01 --data=RoboMimic  --subpolicy_model=ExpWandbLogs/RMP_011/saved_models/Model_epoch5000 --latent_loss_weight=0.0 --z_dimensions=16 --traj_length=-1 --var_skill_length=1 --training_phase_size=2000 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --batch_size=32 --seed=2 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=200 --epochs=5000 --normalization=meanvar --save_freq=500
