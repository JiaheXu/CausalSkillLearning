# Template Baxter Sawyer Run
# python Master.py --name=DJFE_BaxSaw_RMIME_009 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=Roboturk --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RT_001/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=2 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=63 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --target_single_hand=right --model=ExpWandbLogs/DJFE_BaxSaw_RMIME_009/saved_models/Model_epoch8000


# Run from Franka to Roboturk
/ CUDA_VISIBLE_DEVICES=1 python Master.py --name=DJFE_SawyerFranka_023 --train=1 --setting=densityjointfixembedtransfer --data=Roboturk --source_domain=Roboturk --target_domain=RoboMimic --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RT_001/saved_models/Model_epoch500 --target_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=500 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=70 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/Roboturk/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --logdir=/data/tanmayshankar/TrainingLogs/

###################################################
###################################################
# Adapt run from 009.
# Length 47
/ CUDA_VISIBLE_DEVICES=0 python Master.py --name=DJFE_BaxterR_Franka_001 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=RoboMimic --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=47 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --target_single_hand=right --logdir=/data/tanmayshankar/TrainingLogs/

# Length 63
/ CUDA_VISIBLE_DEVICES=0 python Master.py --name=DJFE_BaxterR_Franka_002 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=RoboMimic --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=63 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --target_single_hand=right --logdir=/data/tanmayshankar/TrainingLogs/

########################################### 
###########################################
# Try opposite direction 
/ CUDA_VISIBLE_DEVICES=1 python Master.py --name=DJFE_BaxterR_Franka_003 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=RoboMimic --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --target_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=47 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --source_single_hand=right --logdir=/data/tanmayshankar/TrainingLogs/

# Longer traj
/ python Master.py --name=DJFE_BaxterR_Franka_004 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=RoboMimic --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --target_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=63 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --source_single_hand=right --logdir=/data/tanmayshankar/TrainingLogs/


###################################################
###################################################
# Now for the other hand
###################################################
###################################################
# Adapt run from 009.
# Length 47
/ CUDA_VISIBLE_DEVICES=2 python Master.py --name=DJFE_BaxterL_Franka_001 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=RoboMimic --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=47 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --target_single_hand=left --logdir=/data/tanmayshankar/TrainingLogs/

# Length 63
/ CUDA_VISIBLE_DEVICES=0 python Master.py --name=DJFE_BaxterL_Franka_002 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=RoboMimic --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=63 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --target_single_hand=left --logdir=/data/tanmayshankar/TrainingLogs/

########################################### 
###########################################
# Try opposite direction 
/ CUDA_VISIBLE_DEVICES=3 python Master.py --name=DJFE_BaxterL_Franka_003 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=RoboMimic --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --target_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=47 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --source_single_hand=left --logdir=/data/tanmayshankar/TrainingLogs/

# Longer traj
/ python Master.py --name=DJFE_BaxterL_Franka_004 --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=MIME --target_domain=RoboMimic --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --target_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=10000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=63 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --source_single_hand=left --logdir=/data/tanmayshankar/TrainingLogs/

#####################################################
# Visualizing runs without end effector stuff
/ CUDA_VISIBLE_DEVICES=2 python Master.py --name=DJFE_BaxterL_Franka_001_cleanviz --train=1 --setting=densityjointfixembedtransfer --data=MIME --source_domain=RoboMimic --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --display_freq=500 --discriminator_phase_size=0 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --batch_size=32 --source_model=ExpWandbLogs/RM_007/saved_models/Model_epoch5000 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=0.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=1 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=0 --cross_domain_supervision_loss_weight=0. --normalization=minmax --learning_rate=1e-4 --epochs=2 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=0 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=0 --metric_eval_freq=2000 --cross_domain_density_loss_weight=0. --gmm_variance_value=0.5 --gmm_tuple_variance_value=0.5 --backward_density_loss_weight=1. --forward_density_loss_weight=1. --z_tuple_gmm=1 --cross_domain_z_tuple_density_loss_weight=1.0 --dataset_traj_length_limit=47 --source_datadir=/home/tshankar/Research/Code/Data/Datasets/RoboMimic/ --target_datadir=/home/tshankar/Research/Code/Data/Datasets/MIME/ --forward_tuple_density_loss_weight=1.0 --backward_tuple_density_loss_weight=1.0 --new_supervision=1 --target_single_hand=left --logdir=/data/tanmayshankar/TrainingLogs/ --model=/data/tanmayshankar/TrainingLogs/DJFE_BaxterL_Franka_001/saved_models/Model_epoch10000