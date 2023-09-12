# 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RMOBP_000 --data=RoboMimicObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RMOBP_001 --data=RoboMimicObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/

# Eval
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOBP_000 --data=RoboMimicObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOBP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOBP_001 --data=RoboMimicObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOBP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0

# 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOBP_000_NoR --data=RoboMimicObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOBP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RMOBP_001_NoR --data=RoboMimicObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOBP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0

# Render as video
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RMOBP_000_NoR_Vid --data=RoboMimicObjects --kl_weight=0.0001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --smoothen=0 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RMOBP_001/saved_models/Model_epoch1995 --task_based_shuffling=1 --viz_sim_rollout=0