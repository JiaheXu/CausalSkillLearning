# Generate dataset
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FK_debug --data=FrankaKitchenPreproc --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# Try training
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FK_debug --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# 
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_000 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16

# Debug viz
// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_000_debugviz --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --model=/data/tanmayshankar/TrainingLogs/FKROP_000/saved_models/Model_epoch4000

// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_000_Eval --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --model=/data/tanmayshankar/TrainingLogs/FKROP_000/saved_models/Model_epoch4000

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_001 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21

# Eval
// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_001_Eval --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=4000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21 --model=/data/tanmayshankar/TrainingLogs/FKROP_001/saved_models/Model_epoch4000


# Run for longer
# 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_002 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --model=/data/tanmayshankar/TrainingLogs/FKROP_000/saved_models/Model_epoch4000

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_003 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21  --model=/data/tanmayshankar/TrainingLogs/FKROP_001/saved_models/Model_epoch4000

# Run without KL
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_004 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_005 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21 
 
# Eval
// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_004_Eval --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --model=/data/tanmayshankar/TrainingLogs/FKROP_004/saved_models/Model_epoch3000 --perplexity=10

// MUJOCO_GL='egl' CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=FKROP_005_Eval --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21  --model=/data/tanmayshankar/TrainingLogs/FKROP_005/saved_models/Model_epoch3000 --perplexity=10

# Compute stats
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_stats --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21 

################################
# Run with normalization
################################
# debug
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_debugnorm --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.01 --normalization=minmax


# With KL
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_010 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.01 --normalization=minmax

// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_011 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.01 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21 --normalization=minmax

# No KKL
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_012 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --normalization=minmax

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_013 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21  --normalization=minmax
 
#################################
# Run with meanvar normalization
#################################

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_014 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.01 --normalization=meanvar

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_015 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0.01 --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21 --normalization=meanvar

# No KL
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_016 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --normalization=meanvar

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=FKROP_017 --data=FrankaKitchenRobotObject --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=300 --display_freq=2000 --epochs=10000 --datadir=/data/tanmayshankar/Datasets/FrankaKitchen/ --smoothen=0 --task_based_shuffling=0 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=16 --kl_weight=0. --split_stream_encoder=1 --robot_state_size=9 --env_state_size=21  --normalization=meanvar
 
