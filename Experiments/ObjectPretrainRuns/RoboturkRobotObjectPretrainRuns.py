#######################
# Run on Joint Roboturk Robot Object data.
from locale import normalize


// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTROP_001 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/

// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RTROP_001 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/RTROP_001/saved_models/Model_epoch1500

# Trial with new continuous encoder architecture
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTROP_newarch_trial --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --z_dimensions=16 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/

# Try with factored encoding
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RTROP_newarch_trial --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

# Evaluate
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_001 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_001/saved_models/Model_epoch2000

# Run with diff KL weight
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_002 --data=RoboturkRobotObjects --kl_weight=0.1 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_003 --data=RoboturkRobotObjects --kl_weight=0.01 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

# 
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_Debug --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_001/saved_models/Model_epoch2000

######################
# Rerun with Module dict
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_004 --data=RoboturkRobotObjects --kl_weight=0.0001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_006 --data=RoboturkRobotObjects --kl_weight=0.01 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

# Eval
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400

# Try viz
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400

# Robot_Z viz
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_RobotZViz --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --embedding_visualization_stream='robot'

# Env_Z viz
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_EnvZViz --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --embedding_visualization_stream='env'

#######################
# Debug rollout viz
// CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_debugviz --data=RoboturkRobotObjects --kl_weight=0.01 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --debug=1
/home/tshankar/Research/Code/Data/Datasets/Roboturk/

# Debug rollout viz
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1
# Scale 2
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf5 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1
# 
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf1 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1
#
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf1_rep20 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1
#
# Step rep, Action scaling
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf1_rep10 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1 --sim_viz_step_repetition=10 --sim_viz_action_scale_factor=1.
# Step rep, Action scaling
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf2_rep5 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1 --sim_viz_step_repetition=5 --sim_viz_action_scale_factor=2.
# 
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf2_rep1 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1 --sim_viz_step_repetition=1 --sim_viz_action_scale_factor=2.

// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_sf5_rep1 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1 --sim_viz_step_repetition=1 --sim_viz_action_scale_factor=5.

# 
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_new --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400
#
// CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005_viz1 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400

#
# Now running with new simulation setup.  
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005_sf1pt0_arep20 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=1 --sim_viz_step_repetition=20 --sim_viz_action_scale_factor=1. 

# 
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400 --viz_sim_rollout=0

# FINDING GOOD MODEL SERIES: 
# Test 500
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch500 --viz_sim_rollout=0

# Test 480
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch480 --viz_sim_rollout=0

# Test 510
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch510 --viz_sim_rollout=0

#####################################
    
# Test 500
// CUDA_VISIBLE_DEVICES=0 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch500 --viz_sim_rollout=1 --sim_viz_step_repetition=20 --sim_viz_action_scale_factor=1. 

# Test 480
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch480 --viz_sim_rollout=1 --sim_viz_step_repetition=20 --sim_viz_action_scale_factor=1. 

# Test 510
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch510 --viz_sim_rollout=1 --sim_viz_step_repetition=20 --sim_viz_action_scale_factor=1. 

# Test 480 with video with out sim.
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005_VizVideo --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch480 --viz_sim_rollout=0

# Test 480 with video without sim in old env.
// CUDA_VISIBLE_DEVICES=1 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_005_VizVideo_v03 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch480 --viz_sim_rollout=0




# #######################
# # Retrain with normalization
# CUDA_VISIBLE_DEVICES=0 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_007 --data=RoboturkRobotObjects --kl_weight=0.0001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --normalization=minmax

# CUDA_VISIBLE_DEVICES=2 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_008 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --normalization=minmax

# CUDA_VISIBLE_DEVICES=1 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_009 --data=RoboturkRobotObjects --kl_weight=0.0001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --normalization=meanvar

# CUDA_VISIBLE_DEVICES=3 python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_010 --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --normalization=meanvar


##########################################
##########################################

/ python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_debugenv --data=RoboturkRobotObjects --kl_weight=0.0001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1

##########################################
/ python Master.py --train=1 --setting=pretrain_sub --name=RT_RO_JointSpace_Pre_debugviz --data=RoboturkRobotObjects --kl_weight=0.01 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --smoothen=0 --task_based_shuffling=1 --logdir=/home/tshankar/Research/Code/CausalSkillLearning/Experiments/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --datadir=/home/tshankar/Research/Data/Datasets/Roboturk/


##########################################
CUDA_VISIBLE_DEVICES=2 python Master.py --train=0 --setting=pretrain_sub --name=RT_RO_JointSpace_testviz_restartbach --data=RoboturkRobotObjects --kl_weight=0.001 --var_skill_length=1 --number_layers=4 --hidden_size=48 --batch_size=32 --no_mujoco=1 --seed=0 --epsilon_from=0.3 --epsilon_to=0.1 --epsilon_over=100 --display_freq=2000 --epochs=2000 --datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --smoothen=0 --task_based_shuffling=1 --logdir=/data/tanmayshankar/TrainingLogs/ --z_dimensions=32 --split_stream_encoder=1 --model=/data/tanmayshankar/TrainingLogs/RT_RO_JointSpace_Pre_005/saved_models/Model_epoch400