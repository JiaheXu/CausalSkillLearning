# FixedEmbedding run template
CUDA_VISIBLE_DEVICES=1 python Master.py --name=JFE_MIME_psup_debug --train=1 --setting=jointfixembed --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=2000 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=5 --generator_phase_size=1 --vae_loss_weight=0. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --fix_target=1 --discriminability_weight=1.0 --discriminator_weight=1. --z_transform_discriminator=1 --z_trajectory_discriminability_weight=12. --z_trajectory_discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --identity_translation_loss_weight=0. --number_of_supervised_datapoints=30 --model=ExpWandbLogs/JFE_MIME_psup_046/saved_models/Model_epoch8000 --dataset_traj_length_limit=50

# AsymFixEmbed run template
CUDA_VISIBLE_DEVICES=1 python Master.py --name=JZ_MIME_001 --train=1 --setting=jointtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=5000 --eval_freq=4 --alternating_phase_size=400 --discriminator_phase_size=1 --generator_phase_size=2 --vae_loss_weight=10. --discriminability_weight=1.0 --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --z_transform_discriminator=1 --z_trajectory_discriminability_weight=0. --z_trajectory_discriminator_weight=4. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.05 --epsilon_over=250

# Constructing density based template
CUDA_VISIBLE_DEVICES=1 python Master.py --name=DEN_MIME_debug --train=1 --setting=densityjointtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=2000 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=1 --generator_phase_size=10 --vae_loss_weight=10. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --discriminability_weight=0. --discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --dataset_traj_length_limit=50

# 
CUDA_VISIBLE_DEVICES=1 python Master.py --name=DEN_MIME_debug_plots2 --train=1 --setting=densityjointtransfer --data=MIME --source_domain=MIME --target_domain=MIME --z_dimensions=16 --number_layers=4 --hidden_size=48 --var_number_layers=5 --var_hidden_size=48 --training_phase_size=10 --display_freq=2000 --eval_freq=4 --alternating_phase_size=10 --discriminator_phase_size=1 --generator_phase_size=10 --vae_loss_weight=10. --kl_weight=0.0 --var_skill_length=1 --context=0 --batch_size=32 --source_model=ExpWandbLogs/MJ_111/saved_models/Model_epoch500 --target_model=ExpWandbLogs/MJ_116/saved_models/Model_epoch500 --transfer_learning_rate=1e-4 --fix_source=1 --discriminability_weight=0. --discriminator_weight=1. --eval_transfer_metrics=0 --max_viz_trajs=1 --traj_length=-1 --short_trajectories=1 --epsilon_from=0.3 --epsilon_to=0.01 --epsilon_over=30 --recurrent_translation=0 --cross_domain_supervision=1 --cross_domain_supervision_loss_weight=0.1 --normalization=minmax --learning_rate=1e-4 --epochs=8000 --save_freq=20 --mlp_dropout=0.5 --regularization_weight=0. --no_mujoco=1 --dataset_traj_length_limit=50 --cross_domain_density_loss_weight=1.

