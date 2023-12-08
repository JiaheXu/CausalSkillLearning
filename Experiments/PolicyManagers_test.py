# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from locale import normalize
from os import environ
from headers import *
from PolicyNetworks import *
from RL_headers import *
from PPO_Utilities import PPOBuffer
from Visualizers import BaxterVisualizer, SawyerVisualizer, FrankaVisualizer, ToyDataVisualizer, \
	GRABVisualizer, GRABHandVisualizer, GRABArmHandVisualizer, DAPGVisualizer, \
	RoboturkObjectVisualizer, RoboturkRobotObjectVisualizer,\
	RoboMimicObjectVisualizer, RoboMimicRobotObjectVisualizer, DexMVVisualizer, \
	FrankaKitchenVisualizer, FetchMOMARTVisualizer, DatasetImageVisualizer
	# MocapVisualizer 

# from Visualizers import *
# import TFLogger, DMP, RLUtils
import DMP, RLUtils

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.set_printoptions(sci_mode=False, precision=2)

# Global data list
global global_dataset_list 
global_dataset_list = ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk', \
			'Mocap','OrigRoboMimic','RoboMimic','GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', \
	  		'GRABObject', 'DAPG', 'DAPGHand', 'DAPGObject', 'DexMV', 'DexMVHand', 'DexMVObject', \
			'RoboturkObjects','RoboturkRobotObjects','RoboMimicObjects','RoboMimicRobotObjects', \
			'RoboturkMultiObjets', 'RoboturkRobotMultiObjects', \
			'MOMARTPreproc', 'MOMART', 'MOMARTObject', 'MOMARTRobotObject', 'MOMARTRobotObjectFlat', \
			'FrankaKitchenPreproc', 'FrankaKitchen', 'FrankaKitchenObject', 'FrankaKitchenRobotObject', \
			'RealWorldRigid', 'RealWorldRigidRobot', 'RealWorldRigidJEEF', 'NDAX', 'NDAXMotorAngles']

class PolicyManager_BaseClass():

	def __init__(self):
		super(PolicyManager_BaseClass, self).__init__()

	def setup(self):

		print("RUNNING SETUP OF: ", self)

		# Fixing seeds.
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)
		np.set_printoptions(suppress=True,precision=2)

		self.create_networks()
		self.create_training_ops()
		# self.create_util_ops()
		# self.initialize_gt_subpolicies()

		if self.args.setting=='imitation':
			extent = self.dataset.get_number_task_demos(self.demo_task_index)
		if (self.args.setting=='transfer' and isinstance(self, PolicyManager_Transfer)) or \
			(self.args.setting=='cycle_transfer' and isinstance(self, PolicyManager_CycleConsistencyTransfer)) or \
			(self.args.setting=='fixembed' and isinstance(self, PolicyManager_FixEmbedCycleConTransfer)) or \
			(self.args.setting=='jointtransfer' and isinstance(self, PolicyManager_JointTransfer)) or \
			(self.args.setting=='jointfixembed' and isinstance(self, PolicyManager_JointFixEmbedTransfer)) or \
			(self.args.setting=='jointcycletransfer' and isinstance(self, PolicyManager_JointCycleTransfer)) or \
			(self.args.setting=='jointfixcycle' and isinstance(self, PolicyManager_JointFixEmbedCycleTransfer)) or \
			(self.args.setting=='densityjointtransfer' and isinstance(self, PolicyManager_DensityJointTransfer)) or \
			(self.args.setting=='densityjointfixembedtransfer' and isinstance(self, PolicyManager_DensityJointFixEmbedTransfer)) or \
			(self.args.setting=='iktrainer' and isinstance(self, PolicyManager_IKTrainer)) or \
			(self.args.setting=='downstreamtasktransfer' and isinstance(self, PolicyManager_DownstreamTaskTransfer)):
				extent = self.extent
		else:
			extent = len(self.dataset)-self.test_set_size

		self.index_list = np.arange(0,extent)
		self.initialize_plots()

		# if self.args.setting in ['transfer','cycle_transfer','fixembed','jointtransfer','jointcycletransfer']:
		if self.args.setting in ['jointtransfer'] and isinstance(self, PolicyManager_JointTransfer) or \
			self.args.setting in ['jointfixembed'] and isinstance(self, PolicyManager_JointFixEmbedTransfer) or \
			self.args.setting in ['jointcycletransfer'] and isinstance(self, PolicyManager_JointCycleTransfer) or \
			self.args.setting in ['fixembed'] and isinstance(self, PolicyManager_FixEmbedCycleConTransfer) or \
			self.args.setting in ['jointfixcycle'] and isinstance(self, PolicyManager_JointFixEmbedCycleTransfer) or \
			self.args.setting in ['densityjointtransfer'] and isinstance(self, PolicyManager_DensityJointTransfer) or \
			self.args.setting in ['densityjointfixembedtransfer'] and isinstance(self, PolicyManager_DensityJointFixEmbedTransfer) or \
			self.args.setting in ['downstreamtasktransfer'] and isinstance(self, PolicyManager_DownstreamTaskTransfer):
			self.load_domain_models()

	def initialize_plots(self):
		if self.args.name is not None:
			logdir = os.path.join(self.args.logdir, self.args.name)
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			logdir = os.path.join(logdir, "logs")
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)

		if self.args.data in ['MIME','OldMIME'] and not(self.args.no_mujoco):
			self.visualizer = BaxterVisualizer(args=self.args)
			# self.state_dim = 16
		
		elif (self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']) and not(self.args.no_mujoco):			
			self.visualizer = SawyerVisualizer()
		elif (self.args.data in ['OrigRoboMimic','RoboMimic']) and not(self.args.no_mujoco):			
			self.visualizer = FrankaVisualizer()

		elif self.args.data=='Mocap':
			self.visualizer = MocapVisualizer(args=self.args)
		elif self.args.data in ['GRAB']:
			self.visualizer = GRABVisualizer()
		elif self.args.data in ['GRABHand']:
			self.visualizer = GRABHandVisualizer(args=self.args)
		elif self.args.data in ['GRABArmHand', 'GRABArmHandObject', 'GRABObject']:
			self.visualizer = GRABArmHandVisualizer(args=self.args)		
		elif self.args.data in ['DAPG', 'DAPGHand', 'DAPGObject']:
			self.visualizer = DAPGVisualizer(args=self.args)
		elif self.args.data in ['DexMV', 'DexMVHand', 'DexMVObject']:
			self.visualizer = DexMVVisualizer(args=self.args)
		elif self.args.data in ['RoboturkObjects']:		
			self.visualizer = RoboturkObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboturkRobotObjects']:
			self.visualizer = RoboturkRobotObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicObjects']:
			self.visualizer = RoboMimicObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicRobotObjects']:
			self.visualizer = RoboMimicRobotObjectVisualizer(args=self.args)
		elif self.args.data in ['FrankaKitchenRobotObject']:
			self.visualizer = FrankaKitchenVisualizer(args=self.args)
		elif self.args.data in ['MOMARTRobotObject', 'MOMARTRobotObjectFlat']:			
			if not hasattr(self, 'visualizer'):
				self.visualizer = FetchMOMARTVisualizer(args=self.args)
		elif self.args.data in ['RealWorldRigid', 'NDAX', 'NDAXMotorAngles']:
			self.visualizer = DatasetImageVisualizer(args=self.args)
		else:
			self.visualizer = ToyDataVisualizer()
		

		self.rollout_gif_list = []
		self.gt_gif_list = []

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def collect_inputs(self, i, get_latents=False, special_indices=None, called_from_train=False):	

		if self.args.data=='DeterGoal':
			
			if special_indices is not None:
				i = special_indices

			sample_traj, sample_action_seq = self.dataset[i]
			latent_b_seq, latent_z_seq = self.dataset.get_latent_variables(i)

			start = 0

			if self.args.traj_length>0:
				sample_action_seq = sample_action_seq[start:self.args.traj_length-1]
				latent_b_seq = latent_b_seq[start:self.args.traj_length-1]
				latent_z_seq = latent_z_seq[start:self.args.traj_length-1]
				sample_traj = sample_traj[start:self.args.traj_length]	
			else:
				# Traj length is going to be -1 here. 
				# Don't need to modify action sequence because it does have to be one step less than traj_length anyway.
				sample_action_seq = sample_action_seq[start:]
				sample_traj = sample_traj[start:]
				latent_b_seq = latent_b_seq[start:]
				latent_z_seq = latent_z_seq[start:]

			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 		
			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			# concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)
			# old_concatenated_traj = self.old_concat_state_action(sample_traj, sample_action_seq)
			
			# If the collect inputs function is being called from the train function, 
			# Then we should corrupt the inputs based on how much the input_corruption_noise is set to. 
			# If it's 0., then no corruption. 
			corrupted_sample_action_seq = self.corrupt_inputs(sample_action_seq)
			corrupted_sample_traj = self.corrupt_inputs(sample_traj)

			concatenated_traj = self.concat_state_action(corrupted_sample_traj, corrupted_sample_action_seq)		
			old_concatenated_traj = self.old_concat_state_action(corrupted_sample_traj, corrupted_sample_action_seq)
		
			if self.args.data=='DeterGoal':
				self.conditional_information = np.zeros((self.args.condition_size))
				self.conditional_information[self.dataset.get_goal(i)] = 1
				self.conditional_information[4:] = self.dataset.get_goal_position[i]
			else:
				self.conditional_information = np.zeros((self.args.condition_size))

			if get_latents:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj, latent_b_seq, latent_z_seq
			else:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj

		elif self.args.data in global_data_list:

			# If we're imitating... select demonstrations from the particular task.
			if self.args.setting=='imitation' and \
				 (self.args.data in ['Roboturk','RoboMimic','RoboturkObjects','RoboturkRobotObjects',\
					'RoboMimicObjects','RoboMimicRobotObjects']):
				data_element = self.dataset.get_task_demo(self.demo_task_index, i)
			else:
				data_element = self.dataset[i]

			if not(data_element['is_valid']):
				return None, None, None, None							

			trajectory = data_element['demo']

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
				trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

			action_sequence = np.diff(trajectory,axis=0)

			self.current_traj_len = len(trajectory)

			if self.args.data in ['MIME','OldMIME','GRAB','GRABHand','GRABArmHand', 'GRABArmHandObject', 'GRABObject', 'DAPG', 'DAPGHand', 'DAPGObject', 'DexMV', 'DexMVHand', 'DexMVObject', 'RealWorldRigid']:
				self.conditional_information = np.zeros((self.conditional_info_size))				
			# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic',\
				'RoboMimic','RoboturkObjects','RoboturkRobotObjects', 'RoboMimicObjects', 'RoboMimicRobotObjects']:
				robot_states = data_element['robot-state']
				object_states = data_element['object-state']
				self.current_task_for_viz = data_element['task-id']

				self.conditional_information = np.zeros((self.conditional_info_size))
				# Don't set this if pretraining / baseline.
				if self.args.setting=='learntsub' or self.args.setting=='imitation':
					self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
					self.conditional_information[:,:self.cond_robot_state_size] = robot_states
					# Doing this instead of self.cond_robot_state_size: because the object_states size varies across demonstrations.
					self.conditional_information[:,self.cond_robot_state_size:self.cond_robot_state_size+object_states.shape[-1]] = object_states	
					# Setting task ID too.		
					self.conditional_information[:,-self.number_tasks+data_element['task-id']] = 1.

			# If the collect inputs function is being called from the train function, 
			# Then we should corrupt the inputs based on how much the input_corruption_noise is set to. 
			# If it's 0., then no corruption. 
			corrupted_action_sequence = self.corrupt_inputs(action_sequence)
			corrupted_trajectory = self.corrupt_inputs(trajectory)

			concatenated_traj = self.concat_state_action(corrupted_trajectory, corrupted_action_sequence)		
			old_concatenated_traj = self.old_concat_state_action(corrupted_trajectory, corrupted_action_sequence)

			# # Concatenate
			# concatenated_traj = self.concat_state_action(trajectory, action_sequence)
			# old_concatenated_traj = self.old_concat_state_action(trajectory, action_sequence)

			if self.args.setting=='imitation':
				action_sequence = RLUtils.resample(data_element['demonstrated_actions'],len(trajectory))
				concatenated_traj = np.concatenate([trajectory, action_sequence],axis=1)

			return trajectory, action_sequence, concatenated_traj, old_concatenated_traj, data_element

	def set_extents(self):

		##########################
		# Set extent.
		##########################

		# Modifying to make training functions handle batches. 
		# For every item in the epoch:
		if self.args.setting=='imitation':
			extent = self.dataset.get_number_task_demos(self.demo_task_index)
		# if self.args.setting=='transfer' or self.args.setting=='cycle_transfer' or self.args.setting=='fixembed' or self.args.setting=='jointtransfer':
		if self.args.setting in ['transfer','cycle_transfer','fixembed','jointtransfer','jointcycletransfer','jointfixembed','jointfixcycle','densityjointtransfer','densityjointfixembedtransfer','iktrainer']:
			if self.args.debugging_datapoints>-1:
				extent = self.args.debugging_datapoints
				self.extent = self.args.debugging_datapoints
			else:
				extent = self.extent
		else:
			if self.args.debugging_datapoints>-1:				
				extent = self.args.debugging_datapoints
			else:
				extent = len(self.dataset)-self.test_set_size

		if self.args.task_discriminability or self.args.task_based_supervision:
			extent = self.extent	

		return extent
	
	def train(self, model=None):

		print("Running Main Train Function.")

		########################################
		# (1) Load Model If Necessary
		########################################
		if model:
			print("Loading model in training.")
			self.load_all_models(model)			
		
		########################################
		# (2) Set initial values.
		########################################

		counter = self.args.initial_counter_value
		epoch_time = 0.
		cum_epoch_time = 0.		
		self.epoch_coverage = np.zeros(len(self.dataset))

		########################################
		# (3) Outer loop over epochs. 
		########################################
		
		# For number of training epochs. 
		for e in range(self.number_epochs+1): 
					
			########################################
			# (4a) Bookkeeping
			########################################

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			self.current_epoch_running = e
			print("Starting Epoch: ",e)

			########################################
			# (4b) Set extent of dataset. 
			########################################

			# Modifying to make training functions handle batches. 
			extent = self.set_extents()

			########################################
			# (4c) Shuffle based on extent of dataset. 
			########################################						

			# np.random.shuffle(self.index_list)
			self.shuffle(extent)
			self.batch_indices_sizes = []

			########################################
			# (4d) Inner training loop
			########################################

			t1 = time.time()
			self.coverage = np.zeros(len(self.dataset))

			# For all data points in the dataset. 
			for i in range(0,self.training_extent,self.args.batch_size):				
			# for i in range(0,extent-self.args.batch_size,self.args.batch_size):
				# print("RUN TRAIN", i)
				# Probably need to make run iteration handle batch of current index plus batch size.				
				# with torch.autograd.set_detect_anomaly(True):
				t2 = time.time()

				##############################################
				# (5) Run Iteration
				##############################################

				# print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(self.index_list[i]).zfill(5),"Extent:",extent)
				profile_iteration = 0 				
				if profile_iteration:
					self.lp = LineProfiler()
					self.lp_wrapper = self.lp(self.run_iteration)
					# self.lp_wrapper(counter, self.index_list[i])
					self.lp_wrapper(counter, i)
					self.lp.print_stats()			
				else:													
					# self.run_iteration(counter, self.index_list[i])
					self.run_iteration(counter, i)

				t3 = time.time()
				# print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(self.index_list[i]).zfill(5), "Iter Time:",format(t3-t2,".4f"),"PerET:",format(cum_epoch_time/max(e,1),".4f"),"CumET:",format(cum_epoch_time,".4f"),"Extent:",extent)
				# print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",str(i).zfill(5), "Iter Time:",format(t3-t2,".4f"),"PerET:",format(cum_epoch_time/max(e,1),".4f"),"CumET:",format(cum_epoch_time,".4f"),"Extent:",extent)
				print("Epoch:",e,"Trajectory:",str(i).zfill(5), "Datapoints:",i, "Iter Time:",format(t3-t2,".4f"),"PerET:",format(cum_epoch_time/max(e,1),".4f"),"CumET:",format(cum_epoch_time,".4f"),"Extent:",extent)
				counter = counter+1
				
			##############################################
			# (6) Some more book keeping.
			##############################################
				
			t4 = time.time()
			epoch_time = t4-t1
			cum_epoch_time += epoch_time

			##############################################
			# (7) Automatic evaluation if we need it. 
			##############################################
				
			# if e%self.args.eval_freq==0:
			# 	self.automatic_evaluation(e)

			##############################################
			# (8) Debug
			##############################################
						
			self.epoch_coverage += self.coverage
			# if e%100==0:
			# 	print("Debugging dataset coverage")
			# 	embed()

	def automatic_evaluation(self, e):

		# Writing new automatic evaluation that parses arguments and creates an identical command loading the appropriate model. 
		# Note: If the initial command loads a model, ignore that. 

		command_args = self.args._get_kwargs()			
		base_command = 'python Master.py --train=0 --model={0} --batch_size=1'.format("Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e))

		if self.args.data=='Mocap':
			base_command = './xvfb-run-safe ' + base_command

		# For every argument in the command arguments, add it to the base command with the value used, unless it's train or model. 
		for ar in command_args:
			# Skip model and train, because we need to set these manually.
			if ar[0]=='model' or ar[0]=='train' or ar[0]=='batch_size':
				pass
			# Add the rest
			else:				
				base_command = base_command + ' --{0}={1}'.format(ar[0],ar[1])		
		#  cluster_command = 'python cluster_run.py --partition=learnfair --name={0}_Eval --cmd=\'{1}\''.format(self.args.name, base_command)				

		# NOT RUNNING AUTO EVAL FOR NOW.
		# subprocess.Popen([base_command],shell=True)

	def set_visualizer_object(self):

		#####################################################
		# Set visualizer object. 
		#####################################################
		if self.args.data in ['MIME','OldMIME']:
			self.visualizer = BaxterVisualizer(args=self.args)
			# self.state_dim = 16
		elif (self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']) and not(self.args.no_mujoco):			
			self.visualizer = SawyerVisualizer()
		elif (self.args.data in ['OrigRoboMimic','RoboMimic']) and not(self.args.no_mujoco):			
			self.visualizer = FrankaVisualizer()
		elif self.args.data=='Mocap':
			self.visualizer = MocapVisualizer(args=self.args)
			# Because there are just more invalid DP's in Mocap.
			self.N = 100
		elif self.args.data in ['RoboturkObjects']:
			self.visualizer = RoboturkObjectVisualizer(args=self.args)
		elif self.args.data in ['GRABHand']:
			self.visualizer = GRABHandVisualizer(args=self.args)
			self.N = 200
		elif self.args.data in ['GRABArmHand', 'GRABArmHandObject', 'GRABObject']:
			self.visualizer = GRABArmHandVisualizer(args=self.args)
			self.N = 200
		elif self.args.data in ['DAPG', 'DAPGHand', 'DAPGObject']:
			self.visualizer = DAPGVisualizer(args=self.args)		
		elif self.args.data in ['DexMV', 'DexMVHand', 'DexMVObject']:
			self.visualizer = DexMVVisualizer(args=self.args)		
		elif self.args.data in ['RoboturkRobotObjects']:		
			self.visualizer = RoboturkRobotObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicObjects']:
			self.visualizer = RoboMimicObjectVisualizer(args=self.args)
		elif self.args.data in ['RoboMimicRobotObjects']:
			self.visualizer = RoboMimicRobotObjectVisualizer(args=self.args)			
		elif self.args.data in ['FrankaKitchenRobotObject']:
			self.visualizer = FrankaKitchenVisualizer(args=self.args)
		elif self.args.data in ['MOMARTRobotObject', 'MOMARTRobotObjectFlat']:
			if not hasattr(self, 'visualizer'):
				self.visualizer = FetchMOMARTVisualizer(args=self.args)
		elif self.args.data in ['RealWorldRigid', 'NDAX', 'NDAXMotorAngles']:
			self.visualizer = DatasetImageVisualizer(args=self.args)
		else: 
			self.visualizer = ToyDataVisualizer()

	def per_batch_env_management(self, indexed_data_element):
		
		task_id = indexed_data_element['task-id']
		env_name = self.dataset.environment_names[task_id]
		print("Visualizing a trajectory of task:", env_name)

		self.visualizer.create_environment(task_id=env_name)

	def generate_segment_indices(self, batch_latent_b_torch):
		
		self.batch_segment_index_list = []		

		batch_latent_b = batch_latent_b_torch.detach().cpu().numpy()		

		for b in range(self.args.batch_size):
			segments = np.where(batch_latent_b[:self.batch_trajectory_lengths[b],b])[0]

			# Add last index to segments
			segments = np.concatenate([segments, self.batch_trajectory_lengths[b:b+1]])
			self.batch_segment_index_list.append(segments)
			self.global_segment_index_list.append(segments)
		# Need to perform the same manipulation of segment indices that we did in the forward function call.		

	def visualize_robot_data(self, load_sets=False, number_of_trajectories_to_visualize=None):

		
		if number_of_trajectories_to_visualize is not None:
			self.N = number_of_trajectories_to_visualize
		else:

			####################################
			# TEMPORARILY SET N to 10
			####################################
			# self.N = 33
			self.N = self.args.N_trajectories_to_visualize

		self.rollout_timesteps = self.args.traj_length
	
		self.set_visualizer_object()
		np.random.seed(seed=self.args.seed)

		#####################################################
		# Get latent z sets.
		#####################################################
		
		if not(load_sets):

			#####################################################
			# Select Z indices if necessary.
			#####################################################

			if self.args.split_stream_encoder:
				if self.args.embedding_visualization_stream == 'robot':
					stream_z_indices = np.arange(0,int(self.args.z_dimensions/2))
				elif self.args.embedding_visualization_stream == 'env':
					stream_z_indices = np.arange(int(self.args.z_dimensions/2),self.args.z_dimensions)
				else:
					stream_z_indices = np.arange(0,self.args.z_dimensions)	
			else:
				stream_z_indices = np.arange(0,self.args.z_dimensions)

			#####################################################
			# Initialize variables.
			#####################################################

			# self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
			self.latent_z_set = np.zeros((self.N,len(stream_z_indices)))		
			self.queryjoint_latent_z_set = []
			# These are lists because they're variable length individually.
			self.indices = []
			self.trajectory_set = []
			self.trajectory_rollout_set = []		
			self.rollout_gif_list = []
			self.gt_gif_list = []
			self.task_name_set = []

			#####################################################
			# Create folder for gifs.
			#####################################################

			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
			# Create save directory:
			upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

			if not(os.path.isdir(upper_dir_name)):
				os.mkdir(upper_dir_name)

			self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
			if not(os.path.isdir(self.dir_name)):
				os.mkdir(self.dir_name)
			self.traj_dir_name = os.path.join(self.dir_name, "NumpyTrajs")
			if not(os.path.isdir(self.traj_dir_name)):
				os.mkdir(self.traj_dir_name)
			self.z_dir_name = os.path.join(self.dir_name, "NumpyZs")
			if not(os.path.isdir(self.z_dir_name)):
				os.mkdir(self.z_dir_name)


			self.max_len = 0

			#####################################################
			# Initialize variables.
			#####################################################

			self.shuffle(len(self.dataset)-self.test_set_size, shuffle=True)
			
			self.global_segment_index_list = []

			# print("Embedding before the robot visuals loop.s")
			# embed()

			for j in range(self.N//self.args.batch_size):
				
				number_batches_for_dataset = (len(self.dataset)//self.args.batch_size)+1
				i = j % number_batches_for_dataset

				# (1) Encode trajectory. 
				if self.args.setting in ['learntsub','joint', 'queryjoint']:
					
					
					input_dict, var_dict, eval_dict = self.run_iteration(0, j, return_dicts=True, train=False)
					latent_z = var_dict['latent_z_indices']
					sample_trajs = input_dict['sample_traj']
					data_element = input_dict['data_element']
					latent_b = torch.swapaxes(var_dict['latent_b'], 1,0)

					# Generate segment index list..
					self.generate_segment_indices(latent_b)

					# print("Embed to verify segment indices")
					# embed()

				else:
					print("Running iteration of segment in viz, i: ", i, "j:", j)
					latent_z, sample_trajs, _, data_element = self.run_iteration(0, i, return_z=True, and_train=False)
					# latent_z, sample_trajs, _, data_element = self.run_iteration(0, j*self.args.batch_size, return_z=True, and_train=False)

				if self.args.batch_size>1:

					# Set the max length if it's less than this batch of trajectories. 
					if sample_trajs.shape[0]>self.max_len:
						self.max_len = sample_trajs.shape[0]

					#######################
					# Create env for batch.
					if not(self.args.data in ['RealWorldRigid', 'NDAX', 'NDAXMotorAngles']):
						self.per_batch_env_management(data_element[0])

					for b in range(self.args.batch_size):
						
						self.indices.append(j*self.args.batch_size+b)
						print("#########################################")	
						print("Getting visuals for trajectory: ",j*self.args.batch_size+b)
						# print("Getting visuals for trajectory:")
						# print("j:", j, "b:", b, "j*bs+b:", j*self.args.batch_size+b, "il[j*bs+b]:", self.index_list[j*self.args.batch_size+b] "env:", self.dataset[self.index_list[j*self.args.batch_size+b]]['file'])
						# print("j:", j, "b:", b, "j*bs+b:", j*self.args.batch_size+b, "il[j*bs+b]:", self.index_list[j*self.args.batch_size+b])

						if self.args.setting in ['learntsub','joint','queryjoint']:
							self.latent_z_set[j*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())

							# Rollout each individual trajectory in this batch.
							# trajectory_rollout = self.get_robot_visuals(j*self.args.batch_size+b, latent_z[:,b], sample_trajs[:self.batch_trajectory_lengths[b],b], z_seq=True, indexed_data_element=input_dict['data_element'][b])
							trajectory_rollout = self.get_robot_visuals(j*self.args.batch_size+b, latent_z[:self.batch_trajectory_lengths[b],b], \
												sample_trajs[:self.batch_trajectory_lengths[b],b], z_seq=True, indexed_data_element=input_dict['data_element'][b], \
												segment_indices=self.batch_segment_index_list[b])
							
							self.queryjoint_latent_z_set.append(copy.deepcopy(latent_z[:self.batch_trajectory_lengths[b],b].detach().cpu().numpy()))

							# self.queryjoint_latent_b_set.append(copy.deepcopy(latent_b[:self.batch_trajectory_lengths[b],b].detach().cpu().numpy()))
							
							gt_traj = sample_trajs[:self.batch_trajectory_lengths[b],b]
						else:
							# self.latent_z_set[j*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())
							self.latent_z_set[j*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b,stream_z_indices].detach().cpu().numpy())
			
							# Rollout each individual trajectory in this batch.
							trajectory_rollout = self.get_robot_visuals(j*self.args.batch_size+b, latent_z[0,b], sample_trajs[:,b], indexed_data_element=data_element[b])
							gt_traj = sample_trajs[:,b]
							

						# Now append this particular sample traj and the rollout into trajectroy and rollout sets.
						self.trajectory_set.append(copy.deepcopy(gt_traj))
						self.trajectory_rollout_set.append(copy.deepcopy(trajectory_rollout))
						self.task_name_set.append(data_element[b]['environment-name'])
						#######################
						# Save the GT trajectory, the rollout, and Z into numpy files. 

						#####################################################
						# Save trajectories and Zs
						#####################################################

						k = j*self.args.batch_size+b	
						kstr = str(k).zfill(3)

						# print("Before unnorm")
						# embed()
						if self.args.normalization is not None:
							
							gt_traj = (self.trajectory_set[k] *self.norm_denom_value) + self.norm_sub_value
							gt_traj_tuple = (data_element[b]['environment-name'], gt_traj)
							
							# Don't unnormalize, we already did in get robot visuals. 
							rollout_traj = self.trajectory_rollout_set[k]
							rollout_traj_tuple = (data_element[b]['environment-name'], rollout_traj)
							# rollout_traj = (self.trajectory_rollout_set[k]*self.norm_denom_value) + self.norm_sub_value

						# (trajectory_start * self.norm_denom_value ) + self.norm_sub_value
						# np.save(os.path.join(self.traj_dir_name, "GT_Traj{0}.npy".format(k)), gt_traj)
						# np.save(os.path.join(self.traj_dir_name, "Rollout_Traj{0}.npy".format(k)), rollout_traj)
						# np.save(os.path.join(self.z_dir_name, "Latent_Z{0}.npy".format(k)), self.latent_z_set[k])
												
						np.save(os.path.join(self.traj_dir_name, "Traj{0}_GT.npy".format(kstr)), gt_traj_tuple)
						np.save(os.path.join(self.traj_dir_name, "Traj{0}_Rollout.npy".format(kstr)), rollout_traj_tuple)
						np.save(os.path.join(self.z_dir_name, "Traj{0}_Latent_Z.npy".format(kstr)), self.latent_z_set[k])						

				else:

					print("#########################################")	
					print("Getting visuals for trajectory: ",j,i)

					if latent_z is not None:
						self.indices.append(i)

						if len(sample_trajs)>self.max_len:
							self.max_len = len(sample_trajs)
						# Copy z. 
						self.latent_z_set[j] = copy.deepcopy(latent_z.detach().cpu().numpy())

						trajectory_rollout = self.get_robot_visuals(i, latent_z, sample_trajs)								

						self.trajectory_set.append(copy.deepcopy(sample_trajs))
						self.trajectory_rollout_set.append(copy.deepcopy(trajectory_rollout))	

			# Get MIME embedding for rollout and GT trajectories, with same Z embedding. 
			embedded_z = self.get_robot_embedding()

		#####################################################
		# If precomputed sets.
		#####################################################

		else:

			print("Using Precomputed Latent Set and Embedding.")
			# Instead of computing latent sets, just load them from File path. 
			self.load_latent_sets(self.args.latent_set_file_path)
			
			# print("embedding after load")
			# embed()

			# Get embedded z based on what the perplexity is. 			
			embedded_z = self.embedded_zs.item()["perp{0}".format(int(self.args.perplexity))]
			self.max_len = 0
			for i in range(self.N):
				print("Visualizing Trajectory ", i, " of ",self.N)

				# Set the max length if it's less than this batch of trajectories. 
				if self.gt_trajectory_set[i].shape[0]>self.max_len:
					self.max_len = self.gt_trajectory_set[i].shape[0]			

				dummy_task_id_dict = {}
				dummy_task_id_dict['task-id'] = self.task_id_set[i]
				trajectory_rollout = self.get_robot_visuals(i, self.latent_z_set[i], self.gt_trajectory_set[i], indexed_data_element=dummy_task_id_dict)

			self.indices = range(self.N)


		#####################################################
		# Save the embeddings in HTML files.
		#####################################################

		# print("#################################################")
		# print("Embedding in Visualize robot data in Pretrain PM")
		# print("#################################################")
		# embed()

		gt_animation_object = self.visualize_robot_embedding(embedded_z, gt=True)
		rollout_animation_object = self.visualize_robot_embedding(embedded_z, gt=False)
		
		self.task_name_set_array = np.array(self.task_name_set)

		# Save webpage. 
		self.write_results_HTML()
		
		# Save webpage with plots. 
		self.write_results_HTML(plots_or_gif='Plot')
		
		viz_embeddings = True
		if (self.args.data in ['RealWorldRigid', 'RealWorldRigidRobot']) and (self.args.images_in_real_world_dataset==0):
			viz_embeddings = False

		if viz_embeddings:
			self.write_embedding_HTML(gt_animation_object,prefix="GT")
			self.write_embedding_HTML(rollout_animation_object,prefix="Rollout")

	def preprocess_action(self, action=None):

		########################################
		# Numpy-fy and subsample. (It's 1x|S|, that's why we need to index into first dimension.)

		# It is now 1x|S| or Bx|S|? So squeezing should still be okay... 
		########################################
		
		# action_np = action.detach().cpu().numpy()[0,:8]			
		action_np = action.detach().cpu().squeeze(0).numpy()[...,:8]

		########################################
		# Unnormalize action.
		########################################

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			# Remember. actions are normalized just by multiplying denominator, no addition of mean.
			unnormalized_action = (action_np*self.norm_denom_value)
		else:
			unnormalized_action = action_np
		
		########################################
		# Scale action.
		########################################

		scaled_action = unnormalized_action*self.args.sim_viz_action_scale_factor

		########################################
		# Second unnormalization to undo the visualizer environment normalization.... 
		########################################

		ctrl_range = self.visualizer.environment.sim.model.actuator_ctrlrange
		bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
		weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
		# Modify gripper normalization, so that the env normalization actually happens.
		bias = bias[:-1]
		bias[-1] = 0.
		weight = weight[:-1]
		weight[-1] = 1.
		
		# Unnormalized_scaled_action_for_env_step
		if self.visualizer.new_robosuite:
			unnormalized_scaled_action_for_env_step = scaled_action
		else:
			unnormalized_scaled_action_for_env_step = (scaled_action - bias)/weight

		# print("#####################")
		# print("Vanilla A:", action_np)
		# print("Stat Unnorm A: ", unnormalized_action)
		# print("Scaled A: ", scaled_action)
		# print("Env Unnorm A: ", unnormalized_scaled_action_for_env_step)

		return unnormalized_scaled_action_for_env_step

	def compute_next_state(self, current_state=None, action=None):

		####################################
		# If we're stepping in the environment:
		####################################
		
		if self.args.viz_sim_rollout:
			
			####################################
			# Take environment step.
			####################################

			action_to_execute = self.preprocess_action(action)

			########################################
			# Repeat steps for K times.
			########################################
			
			for k in range(self.args.sim_viz_step_repetition):
				# Use environment to take step.
				env_next_state_dict, _, _, _ = self.visualizer.environment.step(action_to_execute)
				gripper_state = env_next_state_dict[self.visualizer.gripper_key]
				if self.visualizer.new_robosuite:
					joint_state = self.visualizer.environment.sim.get_state()[1][:7]
				else:
					joint_state = env_next_state_dict['joint_pos']

			####################################
			# Assemble robot state.
			####################################
			
			gripper_open = np.array([0.0115, -0.0115])
			gripper_closed = np.array([-0.020833, 0.020833])

			# The state that we want is ... joint state?
			gripper_finger_values = gripper_state
			gripper_values = (gripper_finger_values - gripper_open)/(gripper_closed - gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = 2*finger_diff-1

			########################################
			# Concatenate joint and gripper state. 	
			########################################

			robot_state_np = np.concatenate([joint_state, np.array(gripper_value).reshape((1,))])

			########################################
			# Assemble object state.
			########################################

			# Get just the object pose, object quaternion.
			object_state_np = env_next_state_dict['object-state'][:7]

			########################################
			# Assemble next state.
			########################################

			# Parse next state from dictionary, depending on what dataset we're using.

			# If we're using a dataset with both objects and the robot. 
			if self.args.data in ['RoboturkRobotObjects','RoboMimicRobotObjects']:
				next_state_np = np.concatenate([robot_state_np,object_state_np],axis=0)

			# REMEMBER, We're never actually using an only object dataset here, because we can't actually actuate the objects..
			# # If we're using an object only dataset. 
			# elif self.args.data in ['RoboturkObjects']: 
			# 	next_state_np = object_state_np			

			# If we're using a robot only dataset.
			else:
				next_state_np = robot_state_np

			if self.args.normalization in ['meanvar','minmax']:
				next_state_np = (next_state_np - self.norm_sub_value)/self.norm_denom_value

			# Return torchified version of next_state
			next_state = torch.from_numpy(next_state_np).to(device)	


			# print("embedding at gazoo")
			# embed()
			return next_state, env_next_state_dict[self.visualizer.image_key]

		####################################
		# If not using environment to rollout trajectories.
		####################################

		else:			
			# Simply create next state as addition of current state and action.		
			next_state = current_state+action
			# Return - remember this is already a torch tensor now.
			return next_state, None

	def rollout_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None, z_seq=False, original_trajectory=None):

		rendered_rollout_trajectory = []
		
		if self.args.viz_sim_rollout:

			########################################
			# 0) Reset visualizer environment state. 
			########################################

			self.visualizer.environment.reset()

			# Unnormalize the start state. 
			if self.args.normalization in ['minmax','meanvar']:
				unnormalized_trajectory_start = (trajectory_start * self.norm_denom_value ) + self.norm_sub_value
			else:
				unnormalized_trajectory_start = trajectory_start 
			# Now use unnormalized state to set the trajectory state. 
			self.visualizer.set_joint_pose(unnormalized_trajectory_start)		
			
		########################################
		# 1a) Create placeholder policy input tensor. 
		########################################

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).to(device).float()

		if z_seq:
			subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z[0]).to(device).float()	
		else:
			subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()	

		########################################
		# 1b) Set parameters.
		########################################

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = self.rollout_timesteps-1

		########################################
		# 2) Iterate over rollout length:
		########################################

		for t in range(length):
			
			# print("Pause in rollout")
			# embed()
			current_state = subpolicy_inputs[t,:self.state_dim].clone().detach().cpu().numpy()

			########################################
			# 3) Get action from policy. 
			########################################
			
			# Check if we're visualizing the GT trajectory. 
			if self.args.viz_gt_sim_rollout:
				# If we are, then get the action from the original trajectory, not the policy. 

				# Open loop trajectory execution.
				action_to_execute_ol = torch.from_numpy(original_trajectory[t+1]-original_trajectory[t]).cuda()
				# Closed loop 
				action_to_execute_cl = torch.from_numpy(original_trajectory[t+1]-current_state).cuda()

				action_to_execute = action_to_execute_cl

				print("T:", t, " S:", current_state[:8])
				print("A_ol:", action_to_execute_ol[:8].cpu().numpy())
				print("A_cl:", action_to_execute_cl[:8].cpu().numpy())

				
			else:
				# Assume we always query the policy for actions with batch_size 1 here. 
				actions = self.policy_network.get_actions(subpolicy_inputs, greedy=True, batch_size=1)

				# Select last action to execute. 
				action_to_execute = actions[-1].squeeze(1)

				# Downscale the actions by action_scale_factor.
				action_to_execute = action_to_execute/self.args.action_scale_factor

			########################################
			# 4) Compute next state. 
			########################################
			
			# new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute
			new_state, image = self.compute_next_state(subpolicy_inputs[t,:self.state_dim], action_to_execute)
			rendered_rollout_trajectory.append(image)

			########################################
			# 5) Construct new input row.
			########################################

			# New input row. 
			input_row = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
			input_row[0,:self.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute

			if z_seq:
				input_row[0,2*self.state_dim:] = torch.tensor(latent_z[t+1]).to(device).float()
			else:
				input_row[0,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		# print("Embedding in rollout")
		# embed()
		
		trajectory = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		
		return trajectory, rendered_rollout_trajectory

	def retrieve_unnormalized_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None, z_seq=False, original_trajectory=None):

		trajectory, _ = self.rollout_robot_trajectory(trajectory_start, latent_z, rollout_length=rollout_length, z_seq=z_seq, original_trajectory=original_trajectory)

		return self.unnormalize_trajectory(trajectory)

	def unnormalize_trajectory(self, trajectory):
		# Unnormalize. 
		if self.args.normalization is not None:
			unnormalized_trajectory = (trajectory*self.norm_denom_value) + self.norm_sub_value			
		return unnormalized_trajectory

	def partitioned_rollout_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None, z_seq=False, original_trajectory=None, segment_indices=None):

		# If we're running with a sequential factored encoder network, we have pretrain skill policy.
		# This is only trained to rollout individual skills. 
		# Therefore partition the rollout into components that each only run individual skills. 

		# Set initial start state. Overwrite this later. 
		start_state = copy.deepcopy(trajectory_start)
		rollout_trajectory_segment_list = []

		# For each segment, callout rollout robot trajectory. 
		for k in range(len(segment_indices)-1):

			# Start and end indices are start_index = segment_indices[k], end_index = segment_indices[k+1]
			segment_length = segment_indices[k+1] - segment_indices[k]

			# Technically the latent z should be constant across the segment., so just set it to start value. 
			segment_latent_z = latent_z[segment_indices[k]]

			# Rollout. 
			rollout_trajectory_segment, _ = self.rollout_robot_trajectory(start_state, segment_latent_z, rollout_length=segment_length)
			rollout_trajectory_segment_list.append(copy.deepcopy(rollout_trajectory_segment))

			# Set start state. 
			start_state = copy.deepcopy(rollout_trajectory_segment[-1, :self.state_dim])

		# After having rolled out each component, concatenated the trajectories. 
		rollout_fulltrajectory = np.concatenate(rollout_trajectory_segment_list, axis=0)

		return rollout_fulltrajectory, None

	def get_robot_visuals(self, i, latent_z, trajectory, return_image=False, return_numpy=False, z_seq=False, indexed_data_element=None, segment_indices=None):

		########################################
		# 1) Get task ID. 
		########################################
		# Set task ID if the visualizer needs it. 
		# Set task ID if the visualizer needs it. 
		if indexed_data_element is None or ('task-id' not in indexed_data_element.keys()):
			task_id = None
			env_name = None
		else:			
			if self.args.data in ['NDAX', 'NDAXMotorAngles']:
				task_id = indexed_data_element['task_id']
			else:
				task_id = indexed_data_element['task-id']

			# print("EMBED in grv")
			# embed()
			env_name = self.dataset.environment_names[task_id]
			print("Visualizing a trajectory of task:", env_name)

		########################################
		# 2) Feed Z into policy, rollout trajectory.
		########################################
		
		self.visualizer.create_environment(task_id=env_name)

		if self.args.setting in ['queryjoint']:
			trajectory_rollout, rendered_rollout_trajectory = self.partitioned_rollout_robot_trajectory(trajectory[0], latent_z, rollout_length=max(trajectory.shape[0],0), z_seq=z_seq, original_trajectory=trajectory, segment_indices=segment_indices)
		else:
			trajectory_rollout, rendered_rollout_trajectory = self.rollout_robot_trajectory(trajectory[0], latent_z, rollout_length=max(trajectory.shape[0],0), z_seq=z_seq, original_trajectory=trajectory)

		########################################
		# 3) Unnormalize data. 
		########################################

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			unnorm_gt_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			unnorm_pred_trajectory = (trajectory_rollout*self.norm_denom_value) + self.norm_sub_value
		else:
			unnorm_gt_trajectory = trajectory
			unnorm_pred_trajectory = trajectory_rollout

		if self.args.data == 'Mocap':
			# Get animation object from dataset. 
			animation_object = self.dataset[i]['animation']

		print("We are in the PM visualizer function.")

		# Set task ID if the visualizer needs it. 
		# if indexed_data_element is not None and self.args.data == 'DAPG':
		# 	env_name = indexed_data_element['file']
		# 	print("Visualizing trajectory in task environment:", env_name)
		# elif indexed_data_element is None or ('task_id' not in indexed_data_element.keys()):
		# 	task_id = None
		# 	env_name = None

		if self.args.data=='Mocap':
			# Get animation object from dataset. 
			animation_object = self.dataset[i]['animation']

		# print("We are in the PM visualizer function.")
		# embed()

		########################################
		# 4a) Run unnormalized ground truth trajectory in visualizer. 
		########################################

		##############################
		# ADD CHECK FOR REAL WORLD DATA, and then use dataset image..
		# For now
		##############################

		if self.args.data in ['RealWorldRigid'] and self.args.images_in_real_world_dataset:
			# This should already be segmented to the right start and end point...		
			self.ground_truth_gif = self.visualizer.visualize_prerendered_gif(indexed_data_element['subsampled_images'], gif_path=self.dir_name, gif_name="Traj_{0}_GIF_GT.gif".format(str(i).zfill(3)))
		else:			
			self.ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_GT.gif".format(str(i).zfill(3)), return_and_save=True, end_effector=self.args.ee_trajectories, task_id=env_name)

		# Set plot scaling
		# plot_scale = self.norm_denom_value[6:9].max()
		plot_scale = self.norm_denom_value.max()

		# Also plotting trajectory against time. 
		plt.close()
		# plt.plot(range(unnorm_gt_trajectory.shape[0]),unnorm_gt_trajectory[:,:7])
		# plt.plot(range(unnorm_gt_trajectory.shape[0]),unnorm_gt_trajectory[:,6:9])
		plt.plot(range(unnorm_gt_trajectory.shape[0]),unnorm_gt_trajectory)
		ax = plt.gca()
		ax.set_ylim([-plot_scale, plot_scale])
		plt.savefig(os.path.join(self.dir_name,"Traj_{0}_Plot_GT.png".format(str(i).zfill(3))))
		plt.close()

		########################################
		# 4b) Run unnormalized rollout trajectory in visualizer. 
		########################################

		# Also plotting trajectory against time. 
		plt.close()
		# plt.plot(range(unnorm_pred_trajectory.shape[0]),unnorm_pred_trajectory[:,:7])
		# plt.plot(range(unnorm_pred_trajectory.shape[0]),unnorm_pred_trajectory[:,6:9])
		plt.plot(range(unnorm_pred_trajectory.shape[0]),unnorm_pred_trajectory)
		ax = plt.gca()
		ax.set_ylim([-plot_scale, plot_scale])

		if self.args.viz_sim_rollout:
			# No call to visualizer here means we have to save things on our own. 
			self.rollout_gif = rendered_rollout_trajectory
			
			# Set prefix...
			prefix_list = ['Sim','GTSim']
			gtsim_prefix = prefix_list[self.args.viz_gt_sim_rollout]

			self.visualizer.visualize_prerendered_gif(self.rollout_gif, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_{1}Rollout.gif".format(str(i).zfill(3), gtsim_prefix))
			plt.savefig(os.path.join(self.dir_name,"Traj_{0}_Plot_{1}Rollout.png".format(str(i).zfill(3), gtsim_prefix)))		
		else:
			self.rollout_gif = self.visualizer.visualize_joint_trajectory(unnorm_pred_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_Rollout.gif".format(str(i).zfill(3)), return_and_save=True, end_effector=self.args.ee_trajectories, task_id=env_name)
			
			plt.savefig(os.path.join(self.dir_name,"Traj_{0}_Plot_Rollout.png".format(str(i).zfill(3))))

		plt.close()

		########################################
		# 5) Add to GIF lists. 
		########################################		

		self.gt_gif_list.append(copy.deepcopy(self.ground_truth_gif))
		self.rollout_gif_list.append(copy.deepcopy(self.rollout_gif))
		# print("Embed in get robot viz")
		# embed()
		########################################
		# 6) Return: 
		########################################

		if return_numpy:
			self.ground_truth_gif = np.array(self.ground_truth_gif)
			self.rollout_gif = np.array(self.rollout_gif)

		if return_image:
				return unnorm_pred_trajectory, self.ground_truth_gif, self.rollout_gif
		else:
			return unnorm_pred_trajectory

	def write_results_HTML(self, plots_or_gif='GIF'):
		# Retrieve, append, and print images from datapoints across different models. 

		print("Writing HTML File.")
		# Open Results HTML file. 	    
		with open(os.path.join(self.dir_name,'Results_{0}_{1}.html'.format(self.args.name, plots_or_gif)),'w') as html_file:
			
			# Start HTML doc. 
			html_file.write('<html>')
			html_file.write('<body>')
			html_file.write('<p> Model: {0}</p>'.format(self.args.name))						
			# html_file.write('<p> Average Trajectory Distance: {0}</p>'.format(self.mean_distance))

			extension_dict = {}
			extension_dict['GIF'] = 'gif'
			extension_dict['Plot'] = 'png'

			for i in range(self.N):
				
				if i%100==0:
					print("Datapoint:",i)                        
				html_file.write('<p> <b> Trajectory {}  </b></p>'.format(i))

				file_prefix = self.dir_name

				# Create gif_list by prefixing base_gif_list with file prefix.
				# html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_GT.gif"/>  <img src="Traj_{0}_Rollout.gif"/> </div>'.format(i))
				
				# html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_GIF_GT.gif"/>  <img src="Traj_{0}_GIF_Rollout.gif"/> </div>'.format(i))

				html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_{1}_GT.{2}"/>  <img src="Traj_{0}_{1}_Rollout.{2}"/> </div>'.format(str(i).zfill(3), plots_or_gif, extension_dict[plots_or_gif]))
					
				# Add gap space.
				html_file.write('<p> </p>')

			html_file.write('</body>')
			html_file.write('</html>')

	def write_embedding_HTML(self, animation_object, prefix=""):
		print("Writing Embedding File.")

		t1 = time.time()

		# Adding prefix.
		if self.args.viz_sim_rollout:
			# Modifying prefix based on whether we're visualizing GT Or rollout.
			if self.args.viz_gt_sim_rollout:
				sim_or_not = 'GT_Sim'
			else: 
				sim_or_not = 'Sim'
		else:
			sim_or_not = 'Viz'

		# Open Results HTML file. 	    		
		with open(os.path.join(self.dir_name,'Embedding_{0}_{2}_{1}.html'.format(prefix,self.args.name,sim_or_not)),'w') as html_file:
			
			# Start HTML doc. 
			html_file.write('<html>')
			html_file.write('<body>')
			html_file.write('<p> Model: {0}</p>'.format(self.args.name))

			print("TEMPORARILY EMBEDDING VIA VIDEO RATHER THAN ANIMATION")
			html_file.write(animation_object.to_html5_video())

			###############################
			# Regular embedding as animation
			###############################
			
			# html_file.write(animation_object.to_jshtml())
			# print(animation_object.to_html5_video(), file=html_file)

			html_file.write('</body>')
			html_file.write('</html>')

		t2 = time.time()
		# print("Saving Animation Object.")
		# animation_object.save(os.path.join(self.dir_name,'{0}_Embedding_Video.mp4'.format(self.args.name)))
		# animation_object.save(os.path.join(self.dir_name,'{0}_Embedding_Video.mp4'.format(self.args.name)), writer='imagemagick')
		# t3 = time.time()

		print("Time taken to write this embedding in HTML: ",t2-t1)
		# print("Time taken to save the animation object: ",t3-t2)

	def get_robot_embedding(self, return_tsne_object=False, perplexity=None):

		# # Mean and variance normalize z.
		# mean = self.latent_z_set.mean(axis=0)
		# std = self.latent_z_set.std(axis=0)
		# normed_z = (self.latent_z_set-mean)/std
		normed_z = self.latent_z_set

		if perplexity is None:
			perplexity = self.args.perplexity
		
		print("Perplexity: ", perplexity)

		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=perplexity)
		embedded_zs = tsne.fit_transform(normed_z)

		scale_factor = 1
		scaled_embedded_zs = scale_factor*embedded_zs

		if return_tsne_object:
			return scaled_embedded_zs, tsne
		else:
			return scaled_embedded_zs

	def visualize_robot_embedding(self, scaled_embedded_zs, gt=False):

		# Create figure and axis objects
		# matplotlib.rcParams['figure.figsize'] = [8, 8]
		# zoom_factor = 0.04

		# # Good low res parameters: 
		# matplotlib.rcParams['figure.figsize'] = [8, 8]
		# zoom_factor = 0.04

		# Good spaced out highres parameters: 
		matplotlib.rcParams['figure.figsize'] = [40, 40]			
		# zoom_factor = 0.3
		zoom_factor=0.25

		# Set this parameter to make sure we don't drop frames.
		matplotlib.rcParams['animation.embed_limit'] = 2**128
			
		
		fig, ax = plt.subplots()

		# number_samples = 400
		number_samples = self.N		

		# Create a scatter plot of the embedding itself. The plot does not seem to work without this. 
		ax.scatter(scaled_embedded_zs[:number_samples,0],scaled_embedded_zs[:number_samples,1])
		ax.axis('off')
		ax.set_title("Embedding of Latent Representation of our Model",fontdict={'fontsize':5})
		artists = []
		
		# For number of samples in TSNE / Embedding, create a Image object for each of them. 
		for i in range(len(self.indices)):
			if i%10==0:
				print(i)
			# Create offset image (so that we can place it where we choose), with specific zoom. 

			if gt:
				imagebox = OffsetImage(self.gt_gif_list[i][0],zoom=zoom_factor)
			else:
				imagebox = OffsetImage(self.rollout_gif_list[i][0],zoom=zoom_factor)			

			# Create an annotation box to put the offset image into. specify offset image, position, and disable bounding frame. 
			ab = AnnotationBbox(imagebox, (scaled_embedded_zs[self.indices[i],0], scaled_embedded_zs[self.indices[i],1]), frameon=False)
			# Add the annotation box artist to the list artists. 
			artists.append(ax.add_artist(ab))
			
		def update(t):
			# for i in range(number_samples):
			for i in range(len(self.indices)):
				
				if gt:
					imagebox = OffsetImage(self.gt_gif_list[i][min(t, len(self.gt_gif_list[i])-1)],zoom=zoom_factor)
				else:
					imagebox = OffsetImage(self.rollout_gif_list[i][min(t, len(self.rollout_gif_list[i])-1)],zoom=zoom_factor)			

				ab = AnnotationBbox(imagebox, (scaled_embedded_zs[self.indices[i],0], scaled_embedded_zs[self.indices[i],1]), frameon=False)
				artists.append(ax.add_artist(ab))
			
		# update_len = 20
		print("Maximum length of animation:", self.max_len)
		anim = FuncAnimation(fig, update, frames=np.arange(0, self.max_len), interval=200)

		return anim

	def return_wandb_image(self, image):
		return [wandb.Image(image.transpose(1,2,0))]		

	def return_wandb_gif(self, gif):
		return wandb.Video(gif.transpose((0,3,1,2)), fps=4, format='gif')

	def corrupt_inputs(self, input):
		# 0.1 seems like a good value for the input corruption noise value, that's basically the standard deviation of the Gaussian distribution form which we sample additive noise.
		if isinstance(input, np.ndarray):
			corrupted_input = np.random.normal(loc=0.,scale=self.args.input_corruption_noise,size=input.shape) + input
		else:			
			corrupted_input = torch.randn_like(input)*self.args.input_corruption_noise + input
		return corrupted_input	

	def initialize_training_batches(self):

		print("Initializing batches to manage GPU memory.")
		# Set some parameters that we need for the dry run. 
		extent = len(self.dataset)-self.test_set_size # -self.args.batch_size
		counter = 0
		self.batch_indices_sizes = []
		# self.trajectory_lengths = []
		
		self.current_epoch_running = -1
		print("About to run a dry run. ")
		# Do a dry run of 1 epoch, before we actually start running training. 
		# This is so that we can figure out the batch of 1 epoch.
		 		
		self.shuffle(extent,shuffle=False)		

		# Can now skip this entire block, because we've sorted data according to trajectory length.
		# #########################################################
		# #########################################################
		# for i in range(0,extent,self.args.batch_size):		
		# 	# Dry run iteration. 
		# 	self.run_iteration(counter, self.index_list[i], skip_iteration=True)

		# print("About to find max batch size index.")
		# # Now find maximum batch size iteration. 
		# self.max_batch_size_index = 0
		# self.max_batch_size = 0
		# # traj_lengths = []

		# for x in range(len(self.batch_indices_sizes)):
		# 	if self.batch_indices_sizes[x]['batch_size']>self.max_batch_size:
		# 		self.max_batch_size = self.batch_indices_sizes[x]['batch_size']
		# 		self.max_batch_size_index = self.batch_indices_sizes[x]['i']
		# #########################################################
		# #########################################################

		self.max_batch_size_index = 0
		if self.args.data in ['ToyContext','ContinuousNonZero']:
			self.max_batch_size = 'Full'
		else:
			self.max_batch_size = self.dataset.dataset_trajectory_lengths.max()
								
		print("About to run max batch size iteration.")
		print("This batch size is: ", self.max_batch_size)

		# #########################################################
		# #########################################################
		# # Now run another epoch, where we only skip iteration if it's the max batch size.
		# for i in range(0,extent,self.args.batch_size):
		# 	# Skip unless i is ==max_batch_size_index.
		# 	skip = (i!=self.max_batch_size_index)
		# 	self.run_iteration(counter, self.index_list[i], skip_iteration=skip)
		# #########################################################
		# #########################################################

		# Instead of this clumsy iteration, just run iteration with i=0. 
		self.run_iteration(counter, 0, skip_iteration=0, train=False)

	def task_based_shuffling(self, extent, shuffle=True):
		
		#######################################################################

		# Initialize extent as self.extent
		# extent = self.extent
		index_range = np.arange(0,extent)

		# print("Starting task based shuffling")
		# Implement task ID based shuffling / batching here... 
		self.task_id_map = -np.ones(extent,dtype=int)
		self.task_id_count = np.zeros(self.args.number_of_tasks, dtype=int)		

		for k in range(extent):
			self.task_id_map[k] = self.dataset[k]['task_id']
		for k in range(self.args.number_of_tasks):
			self.task_id_count[k] = (self.task_id_map==k).sum()
		
		# What is this doing?! 
		self.cummulative_count = np.concatenate([np.zeros(1,dtype=int),np.cumsum(self.task_id_count)])

		#######################################################################
		# Now that we have an index map and a count of how many demonstrations there are in each task..
	
		#######################################################################
		# Create blocks. 
		# Best way to perform smart batching is perhaps to sort all indices within a task ID. 
		# Next thing to do is to block up the sorted list.
		# As before, add elements to blocks to ensure it's a full batch.
		#######################################################################
		
		# Get list of indices in each task sorted in decreasing order according to trajectory length for smart batching.
		task_sorted_indices_collection = []			
		for k in range(self.args.number_of_tasks):				
			# task_sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths[self.cummulative_count[k]:self.cummulative_count[k+1]])[::-1]
			task_sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths[self.cummulative_count[k]:self.cummulative_count[k+1]])[::-1]+self.cummulative_count[k]
			task_sorted_indices_collection.append(task_sorted_indices)
		
		# Concatenate this into array. 
		# This allows us to use existing blocking code, and just directly index into this! 
		
		self.concatenated_task_id_sorted_indices = np.concatenate(task_sorted_indices_collection)

		#######################################################################
		# Create blocks..
		#######################################################################

		# Strategy - create blocks from each task ID using task_count, and then just add in more trajectories at random to make it a full batch (if needed).		
		
		self.task_based_shuffling_blocks = []
		self.index_task_id_map = []
		# blocks = []
		task_blocks = []
		counter = 0	

		#######################################################################
		# We're going to create blocks, then pick one of the blocks, maybe based on which bucket the index falls into?
		#######################################################################

		for k in range(self.args.number_of_tasks):
			
			j = 0			 		

			####################################
			# Only try to add an entire batch without resampling if we have more than or exactly enough elements for an entire batch.
			####################################

			while j <= self.task_id_count[k]-self.args.batch_size:
							
				# Add a whole batch.
				block = []

				####################################
				# While we still have items to add to this batch.
				####################################

				while len(block)<self.args.batch_size:				

					# Append index to block.., i.e. TASK SORTED INDEX to block..
					block.append(self.concatenated_task_id_sorted_indices[self.cummulative_count[k]+j])
					j += 1				

				####################################
				# Append this block to the block list. 
				####################################

				if shuffle:
					np.random.shuffle(block)

				self.task_based_shuffling_blocks.append(block)
				self.index_task_id_map.append(k)

			####################################
			# Now that we don't have an entire batch to add. 			
			# Get number of samples we need to add, and check if we need to add at all. 
			####################################

			# If j is ==self.args.batch_size-1, skip this.	
			number_of_samples = self.args.batch_size-(self.task_id_count[k]-j)
			
			# Adding check to ssee if there are actually any elements in this task id... 
			# Otherwise just skip.
			# if number_of_samples>0 and self.task_id_count[k]>0 and number_of_samples<self.args.batch_size:
			if number_of_samples>0 and self.task_id_count[k]>0 and not(j==self.args.batch_size-1):
				# Set pool to sample from. 
				# end_index = -1 if (k+1 >= self.args.number_of_tasks) else k+1
				# random_sample_pool = np.arange(self.cummulative_count[k],self.cummulative_count[end_index])
				random_sample_pool = np.arange(self.cummulative_count[k],self.cummulative_count[k+1])

				samples = np.random.randint(self.cummulative_count[k],high=self.cummulative_count[k+1],size=number_of_samples)
				
				# Create last block. 
				block = []
				# # Add original elements. 
				# [block.append(v) for v in np.arange(self.cummulative_count[k]+j, self.cummulative_count[k+1])]
				# # Now add randomly sampled elements.
				# [block.append(v) for v in samples]

				# Append TASK SORTED INDEX to block..
				# Add original elements. 				
				[block.append(self.concatenated_task_id_sorted_indices[v]) for v in np.arange(self.cummulative_count[k]+j, self.cummulative_count[k+1])]				
				# Now add randomly sampled elements.
				[block.append(self.concatenated_task_id_sorted_indices[v]) for v in samples]

				if shuffle:
					np.random.shuffle(block)

				# Finally append block to block list. 
				self.task_based_shuffling_blocks.append(block)
				self.index_task_id_map.append(k)

		# Also create a block - task ID map.., for easy sampling.. 
		# This is a list of bucket indices for each task that can index into self.task_based_shuffling_blocks...
		self.block_index_list_for_task = []
		
		self.index_task_id_map_array = np.array(self.index_task_id_map)

		for k in range(self.args.number_of_tasks):

			temp_indices = np.where(self.index_task_id_map_array==k)[0]			
			self.block_index_list_for_task.append(temp_indices)	

		# Randomly sample the required number of datapoints. 
		#######################################################################
		# New extent...
		self.extent = len(np.concatenate(self.task_based_shuffling_blocks))
		# Try setting  training extent to same hting...
		self.training_extent = len(np.concatenate(self.task_based_shuffling_blocks))

	def trajectory_length_based_shuffling(self, extent, shuffle=True):
		
		# If we're using full trajectories, do trajectory length based shuffling.
		self.sorted_indices = np.argsort(self.dataset.dataset_trajectory_lengths)[::-1]

		# # Bias towards using shorter trajectories if we're debugging.
		# Use dataset_trajectory_length_bias arg isntaed.
		# if self.args.debugging_datapoints > -1: 
		# 	# BIAS SORTED INDICES AWAY FROM SUPER LONG TRAJECTORIES... 
		# 	self.traj_len_bias = 3000
		# 	self.sorted_indices = self.sorted_indices[self.traj_len_bias:]
		
		# Actually just uses sorted_indices...		
		blocks = [self.sorted_indices[i:i+self.args.batch_size] for i in range(0, extent, self.args.batch_size)]
		
		if shuffle:
			np.random.shuffle(blocks)
		# Shuffled index list is just a flattening of blocks.
		self.index_list = [b for bs in blocks for b in bs]

	def randomized_trajectory_length_based_shuffling(self, extent, shuffle=True):
		
		# Pipline.
		# 0) Set block size, and set extents. 
		# 1) Create sample index list. 
		# 2) Fluff indices upto training_extent size. 
		# 3) Sort based on dataset trajectory length. 
		# 4) Set block size. 
		# 5) Block up. 
		# 6) Shuffle blocks. 
		# 7) Divide blocks. 

		# 0) Set block size, and extents. 
		# The higher the batches per block parameter, more randomness, but more suboptimality in terms of runtime. 
		# With dataset trajectory limit, should not be too bad.  
		batches_per_block = 2

		# Now that extent is set, create rounded down extent.
		self.rounded_down_extent = extent//self.args.batch_size*self.args.batch_size

		# Training extent:
		if self.rounded_down_extent==extent:
			self.training_extent = self.rounded_down_extent
		else:
			# This needs to be done such that we have %3==0 batches. 
			batches_to_add = batches_per_block-(self.rounded_down_extent//self.args.batch_size)%batches_per_block
			self.training_extent = self.rounded_down_extent+self.args.batch_size*batches_to_add

		# 1) Create sample index list. 
		original_index_list = np.arange(0,extent)
		
		# 2) Fluff indices upto training_extent size. 		
		if self.rounded_down_extent==extent:
			index_list = original_index_list
		else:
			# additional_index_list = np.random.choice(original_index_list, size=extent-self.rounded_down_extent, replace=False)			
			additional_index_list = np.random.choice(original_index_list, size=self.training_extent - extent, replace=self.args.replace_samples)			
			index_list = np.concatenate([original_index_list, additional_index_list])		
			
		# 3) Sort based on dataset trajectory length. 
		lengths = self.dataset.dataset_trajectory_lengths[index_list]
		sorted_resampled_indices = np.argsort(lengths)[::-1]

		block_size = batches_per_block * self.args.batch_size

		# 5) Block up, now up till training extent.
		blocks = [index_list[sorted_resampled_indices[i:i+block_size]] for i in range(0, self.training_extent, block_size)]
		# blocks = [sorted_resampled_indices[i:i+block_size] for i in range(0, self.training_extent, block_size)]	
		
		# 6) Shuffle blocks. 
		if shuffle:
			for blk in blocks:			
				np.random.shuffle(blk)

		# 7) Divide blocks. 
		# self.index_list = np.concatenate(blocks)
		self.sorted_indices = np.concatenate(blocks)	

	def random_shuffle(self, extent):

		################################
		# Old block based shuffling.		
		################################
	
		# # Replaces np.random.shuffle(self.index_list) with block based shuffling.
		# index_range = np.arange(0,extent)
		# blocks = [index_range[i:i+self.args.batch_size] for i in range(0, extent, self.args.batch_size)]
		# if shuffle:
		# 	np.random.shuffle(blocks)
		# # Shuffled index list is just a flattening of blocks.
		# self.index_list = [b for bs in blocks for b in bs]

		##########################
		# Set training extents.
		##########################

		# Now that extent is set, create rounded down extent.
		self.rounded_down_extent = extent//self.args.batch_size*self.args.batch_size

		# Training extent:
		if self.rounded_down_extent==extent:
			self.training_extent = self.rounded_down_extent
		else:
			self.training_extent = self.rounded_down_extent+self.args.batch_size

		##########################
		# Now shuffle
		##########################

		original_index_list = np.arange(0,extent)
		if self.rounded_down_extent==extent:
			index_list = original_index_list
		else:

			# print("Debug")
			# embed()

			# additional_index_list = np.random.choice(original_index_list, size=extent-self.rounded_down_extent, replace=False)			
			# additional_index_list = np.random.choice(original_index_list, size=self.training_extent-self.rounded_down_extent, replace=False)
			additional_index_list = np.random.choice(original_index_list, size=self.training_extent-extent, replace=self.args.replace_samples)
			index_list = np.concatenate([original_index_list, additional_index_list])
		np.random.shuffle(index_list)
		self.index_list = index_list

	def shuffle(self, extent, shuffle=True):
	
		realdata = (self.args.data in global_dataset_list)

		# Length based shuffling.
		if isinstance(self, PolicyManager_BatchJoint) or isinstance(self, PolicyManager_IKTrainer):

			print("##############################")
			print("##############################")
			print("Necessarily running randomized traj length based shuffling")
			print("##############################")
			print("##############################")
			# print("About to run trajectory length based shuffling.")
			# self.trajectory_length_based_shuffling(extent=extent,shuffle=shuffle)
			self.randomized_trajectory_length_based_shuffling(extent=extent, shuffle=shuffle)

		# # Task based shuffling.
		# elif self.args.task_discriminability or self.args.task_based_supervision or self.args.task_based_shuffling:
		# 	if isinstance(self, PolicyManager_BatchJoint):						
		# 		if not(self.already_shuffled):
		# 			self.task_based_shuffling(extent=extent,shuffle=shuffle)				
		# 			self.already_shuffled = 1				
			
		# 	# if isinstance(self, PolicyManager_Transfer):
		# 	# Also create an index list to shuffle the order of blocks that we observe...

		# 	# 
		# 	# self.index_list = np.arange(0,extent)				
		# 	# np.random.shuffle(self.index_list)
		# 	self.random_shuffle(extent)

		# Task based shuffling.
		elif self.args.task_discriminability or self.args.task_based_supervision or self.args.task_based_shuffling:						
			self.task_based_shuffling(extent=extent,shuffle=shuffle)							
						
		# Random shuffling.
		else:

			################################
			# Single element based shuffling because datasets are ordered
			################################
			self.random_shuffle(extent)

class PolicyManager_Pretrain(PolicyManager_BaseClass):

	def __init__(self, number_policies=4, dataset=None, args=None):

		if args.setting=='imitation':
			super(PolicyManager_Pretrain, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		else:
			super(PolicyManager_Pretrain, self).__init__()

		self.args = args
		# Fixing seeds.
		print("Setting random seeds.")
		np.random.seed(seed=self.args.seed)
		torch.manual_seed(self.args.seed)	

		self.data = self.args.data
		# Not used if discrete_z is false.
		self.number_policies = number_policies
		self.dataset = dataset

		# Global input size: trajectory at every step - x,y,action
		# Inputs is now states and actions.

		# Model size parameters
		# if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='DirContNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':
		self.state_size = 2
		self.state_dim = 2
		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = 2		
		self.latent_z_dimensionality = self.args.z_dimensions
		self.number_layers = self.args.number_layers
		self.traj_length = 5
		self.number_epochs = self.args.epochs
		self.test_set_size = 500

		stat_dir_name = self.dataset.stat_dir_name
		if self.args.normalization=='meanvar':
			self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
		elif self.args.normalization=='minmax':
			self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			if self.args.data in ['MOMARTRobotObjectFlat']:
				self.norm_denom_value[self.norm_denom_value==0.]=1.

		if self.args.data in ['MIME','OldMIME']:
			self.state_size = 16			
			self.state_dim = 16
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length
			self.number_epochs = self.args.epochs

			if self.args.ee_trajectories:
				if self.args.normalization=='meanvar':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_EE_Mean.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_EE_Var.npy")
				elif self.args.normalization=='minmax':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_EE_Min.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_EE_Max.npy")
			else:
				if self.args.normalization=='meanvar':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_Orig_Mean.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_Orig_Var.npy")
				elif self.args.normalization=='minmax':
					self.norm_sub_value = np.load("Statistics/MIME/MIME_Orig_Min.npy")
					self.norm_denom_value = np.load("Statistics/MIME/MIME_Orig_Max.npy") - self.norm_sub_value

			# Max of robot_state + object_state sizes across all Baxter environments. 			
			self.cond_robot_state_size = 60
			self.cond_object_state_size = 25
			self.test_set_size = 50
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size

		# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
		elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','RoboMimic','OrigRoboMimic']:
			if self.args.gripper:
				self.state_size = 8
				self.state_dim = 8
			else:
				self.state_size = 7
				self.state_dim = 7		
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length

			if self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk']:
				stat_dir_name = "Roboturk"
			elif self.args.data in ['RoboMimic','OrigRoboMimic']:
				stat_dir_name = "Robomimic"
				self.test_set_size = 50

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			# Max of robot_state + object_state sizes across all sawyer environments. 
			# Robot size always 30. Max object state size is... 23. 
			self.cond_robot_state_size = 30			
			self.cond_object_state_size = 23			
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size

		elif self.args.data=='Mocap':
			self.state_size = 22*3
			self.state_dim = 22*3	
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0

		elif self.args.data in ['GRAB']:
			
			self.state_size = 24
			self.state_dim = 24
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['GRABHand']:
			
			self.state_size = 120
			self.state_dim = 120

			if self.args.single_hand in ['left', 'right']:
				self.state_dim //= 2
				self.state_size //= 2

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

			# Modify to zero out for now..
			if self.args.skip_wrist:
				self.norm_sub_value[:3] = 0.
				self.norm_denom_value[:3] = 1.
		
		elif self.args.data in ['GRABArmHand']:
			
			if self.args.position_normalization == 'pelvis':
				self.state_size = 144
				self.state_dim = 144

				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
					self.state_size //= 2
			else:
				self.state_size = 147
				self.state_dim = 147
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['GRABArmHandObject']:
			
			self.state_size = 96
			self.state_dim = 96

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['GRABObject']:
			
			self.state_size = 6
			self.state_dim = 6
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 40
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
		
		elif self.args.data in ['DAPG']:
			
			self.state_size = 51
			self.state_dim = 51
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
		
		elif self.args.data in ['DAPGHand']:
			
			self.state_size = 30
			self.state_dim = 30
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			
		elif self.args.data in ['DAPGObject']:
			
			self.state_size = 21
			self.state_dim = 21
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DAPGFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1

		elif self.args.data in ['DexMV']:
			
			self.state_size = 43
			self.state_dim = 43
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]
		
		elif self.args.data in ['DexMVHand']:
			
			self.state_size = 30
			self.state_dim = 30
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

		elif self.args.data in ['DexMVObject']:
			
			self.state_size = 13
			self.state_dim = 13
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 0
			stat_dir_name = self.args.data

			stat_dir_name = "DexMVFull"			

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
				self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value
				self.norm_denom_value[np.where(self.norm_denom_value==0)] = 1
			self.norm_sub_value = self.norm_sub_value[:self.state_dim]
			self.norm_denom_value = self.norm_denom_value[:self.state_dim]

		elif self.args.data in ['RoboturkObjects','RoboMimicObjects','MOMARTObject']:
			# self.state_size = 14
			# self.state_dim = 14

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 7
			self.state_dim = 7

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50

			# stat_dir_name = "RoboturkObjects"
			# stat_dir_name = self.args.data			

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['RoboturkRobotObjects','RoboMimicRobotObjects']:
			# self.state_size = 14
			# self.state_dim = 14

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 15
			self.state_dim = 15

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50

			# stat_dir_name = "RoboturkRobotObjects"			
			# stat_dir_name = self.args.data

			# if self.args.normalization=='meanvar':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Mean.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Var.npy".format(stat_dir_name))
			# elif self.args.normalization=='minmax':
			# 	self.norm_sub_value = np.load("Statistics/{0}/{0}_Min.npy".format(stat_dir_name))
			# 	self.norm_denom_value = np.load("Statistics/{0}/{0}_Max.npy".format(stat_dir_name)) - self.norm_sub_value

		elif self.args.data in ['RoboturkRobotMultiObjects', 'RoboMimiRobotMultiObjects']:

			# Set state size to 7 for now; because we're not using the relative pose.
			self.state_size = 22
			self.state_dim = 22

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

		elif self.args.data in ['MOMART']:

			self.state_size = 28
			self.state_dim = 28

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

		elif self.args.data in ['MOMARTRobotObject', 'MOMARTRobotObjectFlat']:

			self.state_size = 28
			self.state_dim = 28

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50	

		elif self.args.data in ['MOMARTRobotObjectFlat']:

			self.state_size = 506
			self.state_dim = 506			

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50	

		elif self.args.data in ['FrankaKitchen']:

			self.state_size = 30
			self.state_dim = 30

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

		elif self.args.data in ['FrankaKitchenRobotObject']:

			self.state_size = 30
			self.state_dim = 30

			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length			
			self.conditional_info_size = 0
			self.test_set_size = 50			

			# print("FK Embed")
			# embed()
			
		elif self.args.data in ['RealWorldRigid', 'RealWorldRigidRobot']:

			self.state_size = 21
			self.state_dim = 21

			if self.args.data in ['RealWorldRobotRobot']:
				
				self.state_size = 7
				self.state_dim = 7

				self.norm_sub_value = self.norm_sub_value[:self.state_size]
				self.norm_denom_value = self.norm_denom_value[:self.state_size]
			else:
				#########################################
				# Manually scale.
				#########################################
				
				if self.args.normalization is not None:
					# self.norm_sub_value will remain unmodified. 
					# self.norm_denom_value will get divided by scale.
					self.norm_denom_value /= self.args.state_scale_factor
					# Manually make sure quaternion dims are unscaled.
					self.norm_denom_value[10:14] = 1.
					self.norm_denom_value[17:] = 1.
					self.norm_sub_value[10:14] = 0.
					self.norm_sub_value[17:] = 0.

		elif self.args.data in ['RealWorldRigidJEEF']:

			self.state_size = 28
			self.state_dim = 28

			# self.norm_sub_value will remain unmodified. 
			# self.norm_denom_value will get divided by scale.
			self.norm_denom_value /= self.args.state_scale_factor
			# Manually make sure quaternion dims are unscaled.
			# Now have to do this for EEF, and two objects. 
			self.norm_denom_value[10:14] = 1.
			self.norm_denom_value[17:20] = 1.
			self.norm_denom_value[24:] = 1.
			self.norm_sub_value[10:14] = 0.
			self.norm_sub_value[17:20] = 0.
			self.norm_sub_value[24:] = 0.

		elif self.args.data in ['NDAX']:

			self.state_size = 13
			self.state_dim = 13
			
			# Set orientation dimensions to be unnormalized.
			self.norm_sub_value[10:] = 0.
			self.norm_denom_value[10:] = 1.

		elif self.args.data in ['NDAXMotorAngles']:

			self.state_size = 6
			self.state_dim = 6


			self.norm_denom_value = self.norm_denom_value[:6]
			self.norm_sub_value = self.norm_sub_value[:6]


		elif self.args.data in ['RealWorldRigidHuman']:

			self.state_size = 77
			self.state_dim = 77
			
			# Set orientation dimensions to be unnormalized.
			

		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		self.output_size = self.state_size
		self.traj_length = self.args.traj_length			
		self.conditional_info_size = 0
		self.test_set_size = 0			


		# Training parameters. 		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self. learning_rate = self.args.learning_rate
		
		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_epochs = self.args.epsilon_over
		self.decay_counter = self.decay_epochs*(len(self.dataset)//self.args.batch_size+1)
		self.variance_decay_counter = self.args.policy_variance_decay_over*(len(self.dataset)//self.args.batch_size+1)
		
		if self.args.kl_schedule:
			self.kl_increment_epochs = self.args.kl_increment_epochs
			self.kl_increment_counter = self.kl_increment_epochs*(len(self.dataset)//self.args.batch_size+1)
			self.kl_begin_increment_epochs = self.args.kl_begin_increment_epochs
			self.kl_begin_increment_counter = self.kl_begin_increment_epochs*(len(self.dataset)//self.args.batch_size+1)
			self.kl_increment_rate = (self.args.final_kl_weight-self.args.initial_kl_weight)/(self.kl_increment_counter)
			self.kl_phase_length_counter = self.args.kl_cyclic_phase_epochs*(len(self.dataset)//self.args.batch_size+1)
		# Log-likelihood penalty.
		self.lambda_likelihood_penalty = self.args.likelihood_penalty
		self.baseline = None

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)	
		self.linear_variance_decay_rate = (self.args.initial_policy_variance - self.args.final_policy_variance)/(self.variance_decay_counter)
		self.quadratic_variance_decay_rate = (self.args.initial_policy_variance - self.args.final_policy_variance)/(self.variance_decay_counter**2)

	def create_networks(self):
		
		# print("Embed in create networks")
		# embed()
		
		# Create K Policy Networks. 
		# This policy network automatically manages input size. 
		if self.args.discrete_z:
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).to(device)
		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.latent_z_dimensionality, self.number_layers).to(device)
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

		# Create encoder.
		if self.args.discrete_z: 
			# The latent space is just one of 4 z's. So make output of encoder a one hot vector.		
			self.encoder_network = EncoderNetwork(self.input_size, self.hidden_size, self.number_policies).to(device)
		else:
			# self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality).to(device)

			# if self.args.transformer:
			# 	self.encoder_network = TransformerEncoder(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).to(device)
			# else:

			if self.args.split_stream_encoder:
				self.encoder_network = ContinuousFactoredEncoderNetwork(self.input_size, self.args.var_hidden_size, int(self.latent_z_dimensionality/2), self.args).to(device)
			else:
				self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)
				# self.encoder_network = OldContinuousEncoderNetwork(self.input_size, self.args.var_hidden_size, self.latent_z_dimensionality, self.args).to(device)

		# print("Embed in create networks")
		# embed()

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		self.KLDivergence_loss_function = torch.nn.KLDivLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		if self.args.train_only_policy:
			self.parameter_list = self.policy_network.parameters()
		else:
			self.parameter_list = list(self.policy_network.parameters()) + list(self.encoder_network.parameters())
		
		# Optimize with reguliarzation weight.
		self.optimizer = torch.optim.Adam(self.parameter_list,lr=self.learning_rate,weight_decay=self.args.regularization_weight)

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False, just_subpolicy=False):
		load_object = torch.load(path)

		if self.args.train_only_policy and self.args.train: 		
			self.encoder_network.load_state_dict(load_object['Encoder_Network'])
		else:
			self.policy_network.load_state_dict(load_object['Policy_Network'])
			if not(only_policy):
				self.encoder_network.load_state_dict(load_object['Encoder_Network'])

	def set_epoch(self, counter):
		if self.args.train:

			# Annealing epsilon and policy variance.
			if counter<self.decay_counter:
				self.epsilon = self.initial_epsilon-self.decay_rate*counter
				
				if self.args.variance_mode in ['Constant']:
					self.policy_variance_value = self.args.variance_value
				elif self.args.variance_mode in ['LinearAnnealed']:
					self.policy_variance_value = self.args.initial_policy_variance - self.linear_variance_decay_rate*counter
				elif self.args.variance_mode in ['QuadraticAnnealed']:
					self.policy_variance_value = self.args.final_policy_variance + self.quadratic_variance_decay_rate*((counter-self.variance_decay_counter)**2)				

			else:
				self.epsilon = self.final_epsilon
				if self.args.variance_mode in ['Constant']:
					self.policy_variance_value = self.args.variance_value
				elif self.args.variance_mode in ['LinearAnnealed', 'QuadraticAnnealed']:
					self.policy_variance_value = self.args.final_policy_variance
		else:
			self.epsilon = self.final_epsilon
			# self.policy_variance_value = self.args.final_policy_variance
			
			# Default variance value, but this shouldn't really matter... because it's in test / eval mode.
			self.policy_variance_value = self.args.variance_value
		
		# print("embed in set epoch")
		# embed()

		# Set KL weight. 
		self.set_kl_weight(counter)		

	def set_kl_weight(self, counter):
		
		# Monotonic KL increase.
		if self.args.kl_schedule=='Monotonic':
			if counter>self.kl_begin_increment_counter:
				if (counter-self.kl_begin_increment_counter)<self.kl_increment_counter:
					self.kl_weight = self.args.initial_kl_weight + self.kl_increment_rate*counter
				else:
					self.kl_weight = self.args.final_kl_weight
			else:
				self.kl_weight = self.args.initial_kl_weight

		# Cyclic KL.
		elif self.args.kl_schedule=='Cyclic':

			# Setup is before X epochs, don't decay / cycle. 
			# After X epochs, cycle. 
			
			if counter<self.kl_begin_increment_counter:
				self.kl_weight = self.args.initial_kl_weight				
			else: 			

				
				# While cycling, self.kl_phase_length_counter is the number of iterations over which we repeat. 
				# self.kl_increment_counter is the iterations (within a cycle) over which we increment KL to maximum.
				# Get where in a single cycle it is. 
				kl_counter = counter % self.kl_phase_length_counter

				# If we're done with incremenet, just set to final weight. 
				if kl_counter>self.kl_increment_counter:
					self.kl_weight = self.args.final_kl_weight
				# Otherwise, do the incremene.t 
				else:
					self.kl_weight = self.args.initial_kl_weight + self.kl_increment_rate*kl_counter		
		
		# No Schedule. 
		else:
			self.kl_weight = self.args.kl_weight

		# Adding branch for cyclic KL weight.		

	def visualize_trajectory(self, traj, no_axes=False):

		fig = plt.figure()		
		ax = fig.gca()
		# ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),cmap='jet')
		plt.plot(traj)
		# plt.xlim(-10,10)
		# plt.ylim(-10,10)

		if no_axes:
			plt.axis('off')
		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	def update_plots(self, counter, loglikelihood, sample_traj, stat_dictionary):
		
		# log_dict['Subpolicy Loglikelihood'] = loglikelihood.mean()
		log_dict = {'Subpolicy Loglikelihood': loglikelihood.mean(), 'Total Loss': self.total_loss.mean(), 'Encoder KL': self.encoder_KL.mean(), 'KL Weight': self.kl_weight}
		if self.args.relative_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Relative State Recon Loss'] = self.unweighted_relative_state_reconstruction_loss
			log_dict['Relative State Recon Loss'] = self.relative_state_reconstruction_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.task_based_aux_loss_weight>0.:
			log_dict['Unweighted Task Based Auxillary Loss'] = self.unweighted_task_based_aux_loss
			log_dict['Task Based Auxillary Loss'] = self.task_based_aux_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.relative_state_phase_aux_loss_weight>0.:
			log_dict['Unweighted Relative Phase Auxillary Loss'] = self.unweighted_relative_state_phase_aux_loss
			log_dict['Relative Phase Auxillary Loss'] = self.relative_state_phase_aux_loss
			log_dict['Auxillary Loss'] = self.aux_loss
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Cummmulative Computed State Reconstruction Loss'] = self.unweighted_cummulative_computed_state_reconstruction_loss
			log_dict['Cummulative Computed State Reconstruction Loss'] = self.cummulative_computed_state_reconstruction_loss
		if self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			log_dict['Unweighted Teacher Forced State Reconstruction Loss'] = self.unweighted_teacher_forced_state_reconstruction_loss
			log_dict['Teacher Forced State Reconstruction Loss'] = self.teacher_forced_state_reconstruction_loss
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0. or self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			log_dict['State Reconstruction Loss'] = self.absolute_state_reconstruction_loss

		if counter%self.args.display_freq==0:
			
			if self.args.batch_size>1:
				# Just select one trajectory from batch.
				sample_traj = sample_traj[:,0]

			############
			# Plotting embedding in tensorboard. 
			############

			# Get latent_z set. 
			self.get_trajectory_and_latent_sets(get_visuals=True)

			log_dict['Average Reconstruction Error:'] = self.avg_reconstruction_error

			# Get embeddings for perplexity=5,10,30, and then plot these.
			# Once we have latent set, get embedding and plot it. 
			self.embedded_z_dict = {}
			self.embedded_z_dict['perp5'] = self.get_robot_embedding(perplexity=5)
			self.embedded_z_dict['perp10'] = self.get_robot_embedding(perplexity=10)
			self.embedded_z_dict['perp30'] = self.get_robot_embedding(perplexity=30)

			# Save embedded z's and trajectory and latent sets.
			self.save_latent_sets(stat_dictionary)

			# Now plot the embedding.
			statistics_line = "Epoch: {0}, Count: {1}, I: {2}, Batch: {3}".format(stat_dictionary['epoch'], stat_dictionary['counter'], stat_dictionary['i'], stat_dictionary['batch_size'])
			image_perp5 = self.plot_embedding(self.embedded_z_dict['perp5'], title="Z Space {0} Perp 5".format(statistics_line))
			image_perp10 = self.plot_embedding(self.embedded_z_dict['perp10'], title="Z Space {0} Perp 10".format(statistics_line))
			image_perp30 = self.plot_embedding(self.embedded_z_dict['perp30'], title="Z Space {0} Perp 30".format(statistics_line))
			
			# Now adding image visuals to the wandb logs.
			# log_dict["GT Trajectory"] = self.return_wandb_image(self.visualize_trajectory(sample_traj))
			log_dict["Embedded Z Space Perplexity 5"] = self.return_wandb_image(image_perp5)
			log_dict["Embedded Z Space Perplexity 10"] =  self.return_wandb_image(image_perp10)
			log_dict["Embedded Z Space Perplexity 30"] =  self.return_wandb_image(image_perp30)

		# if counter%self.args.metric_eval_freq==0:
		# 	self.visualize_robot_data(load_sets=False, number_of_trajectories_to_visualize=10)

		wandb.log(log_dict, step=counter)

	def plot_embedding(self, embedded_zs, title, shared=False, trajectory=False):
	
		fig = plt.figure()
		ax = fig.gca()
		
		if shared:
			colors = 0.2*np.ones((2*self.N))
			colors[self.N:] = 0.8
		else:
			colors = 0.2*np.ones((self.N))

		if trajectory:
			# Create a scatter plot of the embedding.

			self.source_manager.get_trajectory_and_latent_sets()
			self.target_manager.get_trajectory_and_latent_sets()

			ratio = 0.4
			color_scaling = 15

			# Assemble shared trajectory set. 
			traj_length = len(self.source_manager.trajectory_set[0,:,0])
			self.shared_trajectory_set = np.zeros((2*self.N, traj_length, 2))
			
			self.shared_trajectory_set[:self.N] = self.source_manager.trajectory_set
			self.shared_trajectory_set[self.N:] = self.target_manager.trajectory_set
			
			color_range_min = 0.2*color_scaling
			color_range_max = 0.8*color_scaling+traj_length-1

			for i in range(2*self.N):
				ax.scatter(embedded_zs[i,0]+ratio*self.shared_trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.shared_trajectory_set[i,:,1],c=colors[i]*color_scaling+range(traj_length),cmap='jet',vmin=color_range_min,vmax=color_range_max)

		else:
			# Create a scatter plot of the embedding.
			ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet')
		
		# Title. 
		ax.set_title("{0}".format(title),fontdict={'fontsize':15})
		fig.canvas.draw()
		# Grab image.
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	def save_latent_sets(self, stats):

		# Save latent sets, trajectory sets, and finally, the embedded z's for later visualization.

		# Create save directory:
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"LatentSetDirectory")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"LatentSetDirectory","E{0}_C{1}".format(stats['epoch'],stats['counter']))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name, "LatentSet.npy") , self.latent_z_set)
		np.save(os.path.join(self.dir_name, "GT_TrajSet.npy") , self.gt_trajectory_set)
		np.save(os.path.join(self.dir_name, "EmbeddedZSet.npy") , self.embedded_z_dict)
		np.save(os.path.join(self.dir_name, "TaskIDSet.npy"), self.task_id_set)

	def load_latent_sets(self, file_path):
		
		self.latent_z_set = np.load(os.path.join(file_path, "LatentSet.npy"))
		self.gt_trajectory_set = np.load(os.path.join(file_path, "GT_TrajSet.npy"), allow_pickle=True)
		self.embedded_zs = np.load(os.path.join(file_path, "EmbeddedZSet.npy"), allow_pickle=True)
		self.task_id_set = np.load(os.path.join(file_path, "TaskIDSet.npy"), allow_pickle=True)

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		if self.args.discrete_z:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sqeuence for policy network. 
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:
			
		
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1)).to(device)

			# Mask input trajectory according to subpolicy dropout. 
			self.subpolicy_input_dropout_layer = torch.nn.Dropout(self.args.subpolicy_input_dropout)

			torch_input_trajectory = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			masked_input_trajectory = self.subpolicy_input_dropout_layer(torch_input_trajectory)
			assembled_inputs[:,:self.input_size] = masked_input_trajectory

			assembled_inputs[range(1,len(input_trajectory)),self.input_size:-1] = latent_z_indices[:-1]
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size:] = latent_z_indices
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sequence for policy network's forward / logprobabilities function. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def get_trajectory_segment(self, i):

		# if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='DirContNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':
		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext','DeterGoal']:
			# Sample trajectory segment from dataset. 
			sample_traj, sample_action_seq = self.dataset[i]

			# Subsample trajectory segment. 		
			start_timepoint = np.random.randint(0,self.args.traj_length-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[start_timepoint:end_timepoint]	
			sample_action_seq = sample_action_seq[start_timepoint:end_timepoint-1]

			self.current_traj_len = self.traj_length

			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

			return concatenated_traj, sample_action_seq, sample_traj
		
		# elif self.args.data in ['MIME','OldMIME','Roboturk','OrigRoboturk','FullRoboturk','Mocap','OrigRoboMimic','RoboMimic']:
	
		elif self.args.data in global_dataset_list:
		
			data_element = self.dataset[i]

			####################################			
			# If Invalid.
			####################################
						
			if not(data_element['is_valid']):
				return None, None, None
			
			####################################
			# Check for gripper.
			####################################
				
			if self.args.gripper:
				trajectory = data_element['demo']
			else:
				trajectory = data_element['demo'][:,:-1]

			####################################
			# If allowing variable skill length, set length for this sample.				
			####################################

			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				# self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
				self.current_traj_len = np.random.choice(np.arange(12,17),p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length

			####################################
			# Sample random start point.
			####################################
			
			if trajectory.shape[0]>self.current_traj_len:

				bias_length = int(self.args.pretrain_bias_sampling*trajectory.shape[0])

				# Probability with which to sample biased segment: 
				sample_biased_segment = np.random.binomial(1,p=self.args.pretrain_bias_sampling_prob)

				# If we want to bias sampling of trajectory segments towards the middle of the trajectory, to increase proportion of trajectory segments
				# that are performing motions apart from reaching and returning. 

				# Sample a biased segment if trajectory length is sufficient, and based on probability of sampling.
				if ((trajectory.shape[0]-2*bias_length)>self.current_traj_len) and sample_biased_segment:		
					start_timepoint = np.random.randint(bias_length, trajectory.shape[0] - self.current_traj_len - bias_length)
				else:
					start_timepoint = np.random.randint(0,trajectory.shape[0]-self.current_traj_len)

				end_timepoint = start_timepoint + self.current_traj_len

				# Get trajectory segment and actions. 
				trajectory = trajectory[start_timepoint:end_timepoint]

				# If normalization is set to some value.
				if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
					trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

				# CONDITIONAL INFORMATION for the encoder... 
				if self.args.data in global_dataset_list:


					pass
				# if self.args.data in ['MIME','OldMIME'] or self.args.data=='Mocap':
				# 	pass
				# elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
				# 	# robot_states = data_element['robot-state'][start_timepoint:end_timepoint]
				# 	# object_states = data_element['object-state'][start_timepoint:end_timepoint]
				# 	pass

				# 	# self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
				# 	# self.conditional_information[:,:self.cond_robot_state_size] = robot_states
				# 	# self.conditional_information[:,self.cond_robot_state_size:object_states.shape[-1]] = object_states								
				# 	# conditional_info = np.concatenate([robot_states,object_states],axis=1)	
			else:					
				return None, None, None

			action_sequence = np.diff(trajectory,axis=0)
			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)

			# NOW SCALE THIS ACTION SEQUENCE BY SOME FACTOR: 
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			return concatenated_traj, scaled_action_sequence, trajectory

	def construct_dummy_latents(self, latent_z):

		if self.args.discrete_z:
			latent_z_indices = latent_z.float()*torch.ones((self.traj_length)).to(device).float()			
		else:
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z.squeeze(0) for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.current_traj_len)).to(device).float()
		####################################
		# why not setting the last one to 1?
		##############
		# latent_b[-1] = 1.

		return latent_z_indices, latent_b			

	def initialize_aux_losses(self):
		
		# Initialize losses.
		self.unweighted_relative_state_reconstruction_loss = 0.
		self.relative_state_reconstruction_loss = 0.
		# 
		self.unweighted_relative_state_phase_aux_loss = 0.
		self.relative_state_phase_aux_loss = 0.
		# 
		self.unweighted_task_based_aux_loss = 0.
		self.task_based_aux_loss = 0.

		# 
		self.unweighted_teacher_forced_state_reconstruction_loss = 0.
		self.teacher_forced_state_reconstruction_loss = 0.
		self.unweighted_cummmulative_computed_state_reconstruction_loss = 0.
		self.cummulative_computed_state_reconstruction_loss = 0.

	def compute_auxillary_losses(self, update_dict):

		self.initialize_aux_losses()

		# Set the relative state reconstruction loss.
		if self.args.relative_state_reconstruction_loss_weight>0.:
			self.compute_relative_state_reconstruction_loss()
		if self.args.task_based_aux_loss_weight>0. or self.args.relative_state_phase_aux_loss_weight>0.:
			self.compute_pairwise_z_distance(update_dict['latent_z'][0])
		# Task based aux loss weight. 
		if self.args.task_based_aux_loss_weight>0.:
			self.compute_task_based_aux_loss(update_dict)
		# Relative. 
		if self.args.relative_state_phase_aux_loss_weight>0.:
			self.compute_relative_state_phase_aux_loss(update_dict)
		if self.args.cummulative_computed_state_reconstruction_loss_weight>0. or self.args.teacher_forced_state_reconstruction_loss_weight>0.:
			self.compute_absolute_state_reconstruction_loss()

		# Weighting the auxillary loss...
		self.aux_loss = self.relative_state_reconstruction_loss + self.relative_state_phase_aux_loss + self.task_based_aux_loss + self.absolute_state_reconstruction_loss

	def compute_pairwise_z_distance(self, z_set):

		# Compute pairwise task based weights.
		self.pairwise_z_distance = torch.cdist(z_set, z_set)[0]

		# Clamped z distance loss. 
		# self.clamped_pairwise_z_distance = torch.clamp(self.pairwise_z_distance - self.args.pairwise_z_distance_threshold, min=0.)
		self.clamped_pairwise_z_distance = torch.clamp(self.args.pairwise_z_distance_threshold - self.pairwise_z_distance, min=0.)

	def compute_relative_state_class_vectors(self, update_dict):

		# Compute relative state vectors.

		# Get original states. 
		robot_traj = self.sample_traj_var[...,:3]
		env_traj = self.sample_traj_var[...,self.args.robot_state_size:self.args.robot_state_size+3]
		relative_state_traj = robot_traj - env_traj		

		# Compute relative state. 
		# relative_state_traj = torch.tensor(robot_traj - env_traj).to(device)
		
		# Compute diff. 
		robot_traj_diff = np.diff(robot_traj, axis=0)
		env_traj_diff = np.diff(env_traj, axis=0)
		relative_state_traj_diff = np.diff(relative_state_traj, axis=0)		

		# Compute norm. 
		robot_traj_norm = np.linalg.norm(robot_traj_diff, axis=-1)
		env_traj_norm = np.linalg.norm(env_traj_diff, axis=-1)
		relative_state_traj_norm = np.linalg.norm(relative_state_traj_diff, axis=-1)

		# Compute sum.
		beta_vector = np.stack([robot_traj_norm.sum(axis=0), env_traj_norm.sum(axis=0), relative_state_traj_norm.sum(axis=0)])

		# Threshold this vector. 
		self.beta_threshold_value = 0.5
		self.thresholded_beta_vector = np.swapaxes((beta_vector>self.beta_threshold_value).astype(float), 0, 1)		
		self.torch_thresholded_beta_vector = torch.tensor(self.thresholded_beta_vector).to(device)

	def compute_task_based_aux_loss(self, update_dict):

		# Task list. 
		task_list = []
		for k in range(self.args.batch_size):
			task_list.append(update_dict['data_element'][k]['task-id'])
		# task_array = np.array(task_list).reshape(self.args.batch_size,1)
		torch_task_array = torch.tensor(task_list, dtype=float).reshape(self.args.batch_size,1).to(device)
		
		# Compute pairwise task based weights. 
		# pairwise_task_matrix = (scipy.spatial.distance.cdist(task_array)==0).astype(int).astype(float)
		pairwise_task_matrix = (torch.cdist(torch_task_array, torch_task_array)==0).int().float()

		# Positive weighted task loss. 
		positive_weighted_task_loss = pairwise_task_matrix*self.pairwise_z_distance

		# Negative weighted task loss. 
		# MUST CHECK SIGNAGE OF THIS. 
		negative_weighted_task_loss = (1.-pairwise_task_matrix)*self.clamped_pairwise_z_distance

		# Total task_based_aux_loss.
		self.unweighted_task_based_aux_loss = (positive_weighted_task_loss + self.args.negative_task_based_component_weight*negative_weighted_task_loss).mean()
		self.task_based_aux_loss = self.args.task_based_aux_loss_weight*self.unweighted_task_based_aux_loss

	def compute_relative_state_phase_aux_loss(self, update_dict):

		# Compute vectors first for the batch.
		self.compute_relative_state_class_vectors(update_dict)

		# Compute similarity of rel state vector across batch.
		self.relative_state_vector_distance = torch.cdist(self.torch_thresholded_beta_vector, self.torch_thresholded_beta_vector)
		self.relative_state_vector_similarity_matrix = (self.relative_state_vector_distance==0).float()
	
		# Now set positive loss.
		positive_weighted_rel_state_phase_loss = self.relative_state_vector_similarity_matrix*self.pairwise_z_distance

		# Set negative component
		negative_weighted_rel_state_phase_loss = (1.-self.relative_state_vector_similarity_matrix)*self.clamped_pairwise_z_distance

		# Total rel state phase loss.
		self.unweighted_relative_state_phase_aux_loss = (positive_weighted_rel_state_phase_loss + self.args.negative_task_based_component_weight*negative_weighted_rel_state_phase_loss).mean()
		self.relative_state_phase_aux_loss = self.args.relative_state_phase_aux_loss_weight*self.unweighted_relative_state_phase_aux_loss

	def compute_relative_state_reconstruction_loss(self):
		
		# Get mean of actions from the policy networks.
		mean_policy_actions = self.policy_network.mean_outputs

		# Get translational states. 
		mean_policy_robot_actions = mean_policy_actions[...,:3]
		mean_policy_env_actions = mean_policy_actions[...,self.args.robot_state_size:self.args.robot_state_size+3]
		# Compute relative actions. 
		mean_policy_relative_state_actions = mean_policy_robot_actions - mean_policy_env_actions

		# Rollout states, then compute relative states - although this shouldn't matter because it's linear. 

		# # Compute relative initial state. 		
		# initial_state = self.sample_traj_var[0]
		# initial_robot_state = initial_state[:,:3]
		# initial_env_state = initial_state[:,self.args.robot_state_size:self.args.robot_state_size+3]
		# relative_initial_state = initial_robot_state - initial_env_state

		# Get relative states.
		robot_traj = self.sample_traj_var[...,:3]
		env_traj = self.sample_traj_var[...,self.args.robot_state_size:self.args.robot_state_size+3]
		relative_state_traj = torch.tensor(robot_traj - env_traj).to(device)
		initial_relative_state = relative_state_traj[0]
		# torch_initial_relative_state = torch.tensor(initial_relative_state).cuda()		

		# Differentiable rollouts. 
		policy_predicted_relative_state_traj = initial_relative_state + torch.cumsum(mean_policy_relative_state_actions, axis=0)

		# Set reconsturction loss.
		self.unweighted_relative_state_reconstruction_loss = (policy_predicted_relative_state_traj - relative_state_traj).norm(dim=2).mean()
		self.relative_state_reconstruction_loss = self.args.relative_state_reconstruction_loss_weight*self.unweighted_relative_state_reconstruction_loss

	def relabel_relative_object_state(self, torch_trajectory):

		# Copy over
		relabelled_state_sequence = torch_trajectory

		# Relabel the dims. 

		print("Debug in Relabel")
		embed()

		torchified_object_state = torch.from_numpy(self.normalized_subsampled_relative_object_state).to(device).view(-1, self.args.batch_size, self.args.env_state_size)		
		relabelled_state_sequence[..., -self.args.env_state_size:] = torchified_object_state

		return relabelled_state_sequence	

	def compute_absolute_state_reconstruction_loss(self):

		# Get the mean of the actions from the policy networks until the penultimate action.
		mean_policy_actions = self.policy_network.mean_outputs[:-1]

		# Initial state - remember, states are Time x Batch x State.
		torch_trajectory = torch.from_numpy(self.sample_traj_var).to(device)

		if self.args.data in ['RealWorldRigidJEEF']:
			torch_trajectory = self.relabel_relative_object_state(torch_trajectory)

		initial_state = torch_trajectory[0]

		# Compute reconstructed trajectory differentiably excluding the first timestep. 
		cummulative_computed_reconstructed_trajectory = initial_state + torch.cumsum(mean_policy_actions, axis=0)
		# Teacher forced state.
		teacher_forced_reconstructed_trajectory = torch_trajectory[:-1] + mean_policy_actions

		# Set both of the reconstruction losses of absolute state.
		self.unweighted_cummulative_computed_state_reconstruction_loss = (cummulative_computed_reconstructed_trajectory - torch_trajectory[1:]).norm(dim=2).mean()
		self.unweighted_teacher_forced_state_reconstruction_loss = (teacher_forced_reconstructed_trajectory - torch_trajectory[1:]).norm(dim=2).mean()
		
		# Weighted losses. 
		self.cummulative_computed_state_reconstruction_loss = self.args.cummulative_computed_state_reconstruction_loss_weight * self.unweighted_cummulative_computed_state_reconstruction_loss
		self.teacher_forced_state_reconstruction_loss = self.args.teacher_forced_state_reconstruction_loss_weight*self.unweighted_teacher_forced_state_reconstruction_loss

		# Merge. 
		self.absolute_state_reconstruction_loss = self.cummulative_computed_state_reconstruction_loss + self.teacher_forced_state_reconstruction_loss

	def update_policies_reparam(self, loglikelihood, encoder_KL, update_dict=None):
		
		self.optimizer.zero_grad()

		# Losses computed as sums.
		# self.likelihood_loss = -loglikelihood.sum()
		# self.encoder_KL = encoder_KL.sum()

		# Instead of summing losses, we should try taking the mean of the  losses, so we can avoid running into issues of variable timesteps and stuff like that. 
		# We should also consider training with randomly sampled number of timesteps.
		self.likelihood_loss = -loglikelihood.mean()
		self.encoder_KL = encoder_KL.mean()

		self.compute_auxillary_losses(update_dict)
		# Adding a penalty for link lengths. 
		# self.link_length_loss = ... 

		self.total_loss = (self.likelihood_loss + self.kl_weight*self.encoder_KL + self.aux_loss) 
		# + self.link_length_loss) 

		if self.args.debug:
			print("Embedding in Update subpolicies.")
			embed()

		self.total_loss.backward()
		self.optimizer.step()

	def rollout_visuals(self, i, latent_z=None, return_traj=False, rollout_length=None, traj_start=None):

		# Initialize states and latent_z, etc. 
		# For t in range(number timesteps):
		# 	# Retrieve action by feeding input to policy. 
		# 	# Step in environment with action.
		# 	# Update inputs with new state and previously executed action. 

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.state_dim = 2
			self.rollout_timesteps = 5
		elif self.args.data in ['MIME','OldMIME']:
			self.state_dim = 16
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
			self.state_dim = 8
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRAB']:
			self.state_dim = 24
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABArmHand']:
			if self.args.position_normalization == 'pelvis':
				self.state_dim = 144
				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
			else:
				self.state_dim = 147
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABArmHandObject']:
			self.state_size = 96
			self.state_dim = 96
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABObject']:
			self.state_dim = 6
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['GRABHand']:
			self.state_dim = 120
			if self.args.single_hand in ['left', 'right']:
				self.state_dim //= 2
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPG']:
			self.state_dim = 51
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPGHand']:
			self.state_dim = 30
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DAPGObject']:
			self.state_dim = 21
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMV']:
			self.state_dim = 43
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMVHand']:
			self.state_dim = 30
			self.rollout_timesteps = self.traj_length
		elif self.args.data in ['DexMVObject']:
			self.state_dim = 13
			self.rollout_timesteps = self.traj_length

		if rollout_length is not None:
			self.rollout_timesteps = rollout_length
	
		if traj_start is None:
			start_state = torch.zeros((self.state_dim))
		else:
			start_state = torch.from_numpy(traj_start)
		

		if self.args.discrete_z:
			# Assuming 4 discrete subpolicies, just set subpolicy input to 1 at the latent_z index == i. 
			subpolicy_inputs = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
			subpolicy_inputs[0,self.input_size+i] = 1. 
		else:
			subpolicy_inputs = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[0,self.input_size:] = latent_z

		subpolicy_inputs[0,:self.state_dim] = start_state
		# subpolicy_inputs[0,-1] = 1.		
		
		for t in range(self.rollout_timesteps-1):

			actions = self.policy_network.get_actions(subpolicy_inputs,greedy=True,batch_size=1)
			
			# Select last action to execute.
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor

			# Compute next state. 
			new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute

			# New input row: 
			if self.args.discrete_z:
				input_row = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
				input_row[0,self.input_size+i] = 1. 
			else:
				input_row = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device).float()
				input_row[0,self.input_size:] = latent_z
			input_row[0,:self.state_dim] = new_state
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute	
			# input_row[0,-1] = 1.

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)
		# print("latent_z:",latent_z)
		trajectory_rollout = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		# print("Trajectory:",trajectory_rollout)

		if return_traj:
			return trajectory_rollout		

	def run_iteration(self, counter, i, return_z=False, and_train=True):

		####################################
		####################################
		# Basic Training Algorithm: 
		# For E epochs:
		# 	# For all trajectories:
		#		# Sample trajectory segment from dataset. 
		# 		# Encode trajectory segment into latent z. 
		# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		# 		# Update parameters. 
		####################################
		####################################

		self.set_epoch(counter)

		############# (0) ##################
		# Sample trajectory segment from dataset. 
		####################################

		# Sample trajectory segment from dataset.
		input_dict = {}

		input_dict['state_action_trajectory'], input_dict['sample_action_seq'], input_dict['sample_traj'], input_dict['data_element'] = self.get_trajectory_segment(i)
		# state_action_trajectory, sample_action_seq, sample_traj, data_element  = self.get_trajectory_segment(i)
		# self.sample_traj_var = sample_traj
		self.sample_traj_var = input_dict['sample_traj']
		self.input_dict = input_dict
		####################################
		############# (0a) #############
		####################################

		# Corrupt the inputs according to how much input_corruption_noise is set to.
		state_action_trajectory = self.corrupt_inputs(input_dict['state_action_trajectory'])

		if state_action_trajectory is not None:
			
			####################################
			############# (1) #############
			####################################

			torch_traj_seg = torch.tensor(state_action_trajectory).to(device).float()
			# Encode trajectory segment into latent z. 		
						
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon)
			# latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg)
			
			####################################
			########## (2) & (3) ##########
			####################################

			# print("Embed in rut iter")
			# embed()

			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

			############# (3a) #############
			_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(state_action_trajectory, latent_z_seq, latent_b, input_dict['sample_action_seq'])
			
			############# (3b) #############
			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)

			loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq, self.policy_variance_value)
			loglikelihood = loglikelihoods[:-1].mean()
			 
			if self.args.debug:
				print("Embedding in Train.")
				embed()

			####################################
			# (4) Update parameters. 
			####################################
			
			if self.args.train and and_train:

				####################################
				# (4a) Update parameters based on likelihood, subpolicy inputs, and kl divergence.
				####################################
				
				update_dict = input_dict
				update_dict['latent_z'] = latent_z				

				self.update_policies_reparam(loglikelihood, kl_divergence, update_dict=update_dict)

				####################################
				# (4b) Update Plots. 
				####################################
				
				stats = {}
				stats['counter'] = counter
				stats['i'] = i
				stats['epoch'] = self.current_epoch_running
				stats['batch_size'] = self.args.batch_size			
				self.update_plots(counter, loglikelihood, state_action_trajectory, stats)

				####################################
				# (5) Return.
				####################################

			if return_z:
				return latent_z, input_dict['sample_traj'], sample_action_seq, input_dict['data_element']
									
		else: 
			return None, None, None

	def evaluate_metrics(self):		
		self.distances = -np.ones((self.test_set_size))
		self.robot_z_nn_distances = -np.ones((self.test_set_size))
		self.env_z_nn_distances = -np.ones((self.test_set_size))
		# Get test set elements as last (self.test_set_size) number of elements of dataset.
		for i in range(self.test_set_size):

			index = i + len(self.dataset)-self.test_set_size
			print("Evaluating ", i, " in test set, or ", index, " in dataset.")
			# Get latent z. 					
			latent_z, sample_traj, sample_action_seq, _ = self.run_iteration(0, index, return_z=True)

			if sample_traj is not None:

				# Feed latent z to the rollout.
				# rollout_trajectory = self.rollout_visuals(index, latent_z=latent_z, return_traj=True)
				rollout_trajectory, rendered_rollout_trajectory = self.rollout_robot_trajectory(sample_traj[0], latent_z, rollout_length=len(sample_traj))

				self.distances[i] = ((sample_traj-rollout_trajectory)**2).mean()	

			robot_nn_z_dist, env_nn_z_dist = self.evaluate_z_distances_for_batch(latent_z)

		self.mean_distance = self.distances[self.distances>0].mean()

	def evaluate(self, model=None, suffix=None):

		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
			self.visualize_embedding_space(suffix=suffix)

		if self.args.data in global_dataset_list:

			print("Running Evaluation of State Distances on small test set.")
			# self.evaluate_metrics()		

			# Only running viz if we're actually pretraining.
			if self.args.traj_segments:
				print("Running Visualization on Robot Data.")	

				# self.visualize_robot_data(load_sets=True)
				whether_load_z_set = self.args.latent_set_file_path is not None

				# print("###############################################")
				# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				# print("Temporarily not visualizing.")
				# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				# print("###############################################")
				self.visualize_robot_data(load_sets=whether_load_z_set)

				
				print("###############################################")
				print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				print("Query before we run get trajectory latent sets, so latent_z_set isn't overwritten..")
				print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
				print("###############################################")				
				embed()

				# Get reconstruction error... 
				self.get_trajectory_and_latent_sets(get_visuals=True)
				print("The Average Reconstruction Error is: ", self.avg_reconstruction_error)


			else:
				# Create save directory:
				upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

				if not(os.path.isdir(upper_dir_name)):
					os.mkdir(upper_dir_name)

				model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))				
				self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))

				if not(os.path.isdir(self.dir_name)):
					os.mkdir(self.dir_name)

	def get_trajectory_and_latent_sets(self, get_visuals=True):
		# For N number of random trajectories from MIME: 
		#	# Encode trajectory using encoder into latent_z. 
		# 	# Feed latent_z into subpolicy. 
		#	# Rollout subpolicy for t timesteps. 
		#	# Plot rollout.
		# Embed plots. 

		# Set N:
		self.N = 500

		self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))
			
		if self.args.setting=='transfer' or self.args.setting=='cycle_transfer' or self.args.setting=='fixembed':
			if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext']:
				self.state_dim = 2
				self.rollout_timesteps = 5		
			if self.args.data in ['MIME','OldMIME']:
				self.state_dim = 16
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['Roboturk','OrigRoboturk','FullRoboturk','OrigRoboMimic','RoboMimic']:
				self.state_dim = 8
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRAB']:
				self.state_dim = 24
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABArmHand']:
				if self.args.position_normalization == 'pelvis':
					self.state_dim = 144
					if self.args.single_hand in ['left', 'right']:
						self.state_dim //= 2
				else:
					self.state_dim = 147
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABArmHandObject']:
				self.state_dim = 96
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABObject']:
				self.state_dim = 6
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['GRABHand']:
				self.state_dim = 126
				if self.args.single_hand in ['left', 'right']:
					self.state_dim //= 2
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DAPG']:
				self.state_dim = 51
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DAPGHand']:
				self.state_dim = 30
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DAPGObject']:
				self.state_dim = 21
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DexMV']:
				self.state_dim = 43
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DexMVHand']:
				self.state_dim = 30
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['DexMVObject']:
				self.state_dim = 13
				self.rollout_timesteps = self.traj_length
			if self.args.data in ['RoboturkObjects']:
				# Now switching to using 7 dimensions instead of 14, so as to not use relative pose.
				self.state_dim = 7
				# self.state_dim = 14
				self.rollout_timesteps = self.traj_length

			self.trajectory_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))

		else:
			self.trajectory_set = []
		# self.gt_trajectory_set = np.zeros((self.N, self., self.state_dim))
		
		self.gt_trajectory_set = []
		# Save TASK IDs 
		self.task_id_set = []

		# Use the dataset to get reasonable trajectories (because without the information bottleneck / KL between N(0,1), cannot just randomly sample.)
		# # for i in range(self.N//self.args.batch_size+1, 32)
		# for i in range(0, self.N, self.args.batch_size):
		for i in range(self.N//self.args.batch_size+1):

			# Mapped index
			number_batches_for_dataset = (len(self.dataset)//self.args.batch_size)+1
			j = i % number_batches_for_dataset

			########################################
			# (1) Encoder trajectory. 
			########################################

			latent_z, sample_trajs, _, data_element = self.run_iteration(0, j*self.args.batch_size, return_z=True, and_train=False)

			########################################
			# Iterate over items in the batch.
			########################################
			# print("Embed in latent set creation")
			# embed()

			for b in range(self.args.batch_size):

				if self.args.batch_size*i+b>=self.N:
					break

				self.latent_z_set[i*self.args.batch_size+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())
				# self.latent_z_set[i+b] = copy.deepcopy(latent_z[0,b].detach().cpu().numpy())
				self.gt_trajectory_set.append(copy.deepcopy(sample_trajs[:,b]))
				
				self.task_id_set.append(data_element[b]['task-id'])

				if get_visuals:
					# (2) Now rollout policy.	
					if self.args.setting=='transfer' or self.args.setting=='cycle_transfer' or self.args.setting=='fixembed':
						self.trajectory_set[i*self.args.batch_size+b] = self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True, traj_start=sample_trajs[0,b])
						# self.trajectory_set[i+b] = self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True)
					elif self.args.setting=='pretrain_sub':							
						self.trajectory_set.append(self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True, traj_start=sample_trajs[0,b], rollout_length=sample_trajs.shape[0]))
					else:
						self.trajectory_set.append(self.rollout_visuals(i, latent_z=latent_z[0,b], return_traj=True, traj_start=sample_trajs[0,b]))
			
			if self.args.batch_size*i+b>=self.N:
				break

		# print("Embed in latent set creation before trajectory error evaluation.")
		# embed()

		# Compute average reconstruction error.
		if get_visuals:
			self.gt_traj_set_array = np.array(self.gt_trajectory_set, dtype=object)
			self.trajectory_set = np.array(self.trajectory_set, dtype=object)

			# self.gt_traj_set_array = np.array(self.gt_trajectory_set)
			# self.trajectory_set = np.array(self.trajectory_set)

			# self.avg_reconstruction_error = (self.gt_traj_set_array-self.trajectory_set).mean()
			self.reconstruction_errors = np.zeros(len(self.gt_traj_set_array))
			for k in range(len(self.reconstruction_errors)):
				self.reconstruction_errors[k] = ((self.gt_traj_set_array[k]-self.trajectory_set[k])**2).mean()
			self.avg_reconstruction_error = self.reconstruction_errors.mean()
		else:
			self.avg_reconstruction_error = 0.

	def visualize_embedding_space(self, suffix=None):

		self.get_trajectory_and_latent_sets()

		# TSNE on latentz's.
		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=self.args.perplexity)
		embedded_zs = tsne.fit_transform(self.latent_z_set)

		# ratio = 0.3
		# if self.args.setting in ['transfer','cycletransfer','']
		ratio = (embedded_zs.max()-embedded_zs.min())*0.01
		
		for i in range(self.N):
			plt.scatter(embedded_zs[i,0]+ratio*self.trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.trajectory_set[i,:,1],c=range(self.rollout_timesteps),cmap='jet')

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))		
		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		if suffix is not None:
			self.dir_name = os.path.join(self.dir_name, suffix)

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		# Format with name.
		plt.savefig("{0}/Embedding_Joint_{1}.png".format(self.dir_name,self.args.name))
		plt.close()	

	def create_z_kdtrees(self):
		
		####################################
		# Algorithm to construct models
		####################################

		# 0) Assume that the encoder(s) are trained, and that the latent space is trained.
		# 1) Maintain map of Z_R <--> Z_E. 
		# 	1a) Check that the latent_z_sets are tuples. 
		# 	1b) Seems like we don't actually need the map if the latent z sets are constructed by the same function / indexing.
		# 2) Construct KD Trees. 
		# 	2a) KD_R = KDTREE( {Z_R} )
		# 	2b) KD_E = KDTREE( {Z_E} )
		
		# self.kdtree_robot_z = KDTree(self.robot_latent_z_set)
		# self.kdtree_env_z = KDTree(self.env_latent_z_set)		

		self.kdtree_dict = {}
		self.kdtree_dict['robot'] = KDTree(self.stream_latent_z_dict['robot'])
		self.kdtree_dict['env'] = KDTree(self.stream_latent_z_dict['env'])

	def get_query_trajectory(self, input_state_trajectory, stream=None):
		
		# Assume trajectory is dimensions |T| x |S|. 
		index_dict = {}
		index_dict['robot'] = np.arange(0,8)
		index_dict['env'] = np.arange(8,15)

		indices = index_dict[stream] 
		
		stream_input_state_trajectory = input_state_trajectory[:,indices]
		
		# Get actions. 
		actions = np.diff(stream_input_state_trajectory, axis=0)

		# Pad actions.
		padded_actions = np.concatenate([actions,np.zeros((1,stream_input_state_trajectory.shape[1]))], axis=0)

		# Concatenate state and actions. 
		state_action_traj = np.concatenate([stream_input_state_trajectory, padded_actions], axis=1)

		# Torchify. 
		torch_state_action_traj = torch.from_numpy(state_action_traj).to(device).float()

		return torch_state_action_traj

	def retrieve_nearest_neighbors_from_trajectory(self, trajectory, stream, number_neighbors=1, artificial_batch_size=1):

		# Based on stream, set which KDTree and which latent set to use. 
		kdtree = self.kdtree_dict[stream]
		latent_set = self.stream_latent_z_dict[stream]
		if stream=='robot':
			net_dict = self.encoder_network.robot_network_dict
			size_dict = self.encoder_network.robot_size_dict
		elif stream=='env':
			net_dict = self.encoder_network.env_network_dict
			size_dict = self.encoder_network.env_size_dict

		####################################
		# Query, given a trajectory
		####################################

		# 0) Assumes that we have a concatenation of states and actions.
		# 1) z_r = E_r (Tau_r) || z_e = E_e (Tau_e)
		# 2) z_r^{*NN} = KDT_r.query(z_r) || z_e^{*NN} = KDT_e.query(z_e) 	

		# 1) Query encoder for latent representation of trajectory.
		# Well, we just run super.forward() of the encoder network, which is a continuous factored encoder
		# which inherits its forward function from the continuous encoder. 
		
		# Do not need epsilon or eval. 
		retrieved_z, _, _, _ = self.encoder_network.run_super_forward(trajectory, epsilon=0.0, \
			# network_dict=self.encoder_network.robot_network_dict, size_dict=self.encoder_network.robot_size_dict, artificial_batch_size=1)
			network_dict=net_dict, size_dict=size_dict, artificial_batch_size=artificial_batch_size)
		
		# 2) Query KD Tree with encoding of given trajectory. 
		z_neighbor_distances, z_neighbors_indices = kdtree.query(retrieved_z.detach().cpu().numpy(), k=number_neighbors)

		return z_neighbor_distances, z_neighbor_indices

	def retrieve_cross_indexed_nearest_neighbor_from_trajectory(self, trajectory, stream, number_neighbors=1):

		# Get neighbors. 
		z_neighbor_distances , z_neighbor_indices = self.retrieve_nearest_neighbors_from_trajectory(trajectory, stream, number_neighbors)

		# Cross index. 				
		cross_stream = set(self.stream_latent_z_dict.keys()) - set([stream])
		cross_latent_set = self.stream_latent_z_dict[cross_stream]
		# 	desired_z_e = self.robot_latent_z_set[z_r_nearest_neighbor_index]

		cross_indexed_z = cross_latent_set[z_neighbor_indices]

		return cross_indexed_z
	
	def define_forward_inverse_models(self):

		# 0) Get latent z sets.
		# 1) Create KD trees. 

		# 0) Make sure we've run visualize_robot_data; then split z sets. 
		# Run visualize_robot_data. 

		# # Create z sets. 
		# self.robot_latent_z_set = copy.deepcopy(self.latent_z_set[:,:int(self.latent_z_dimensionality/2)])
		# self.env_latent_z_set = copy.deepcopy(self.latent_z_set[:,int(self.latent_z_dimensionality/2):])

		# Single stream latent z set dict. 
		self.stream_latent_z_dict = {}
		self.stream_latent_z_dict['robot']= copy.deepcopy(self.latent_z_set[:,:int(self.latent_z_dimensionality/2)])
		self.stream_latent_z_dict['env']= copy.deepcopy(self.latent_z_set[:,int(self.latent_z_dimensionality/2):])
		
		# 1) Create KD Trees.
		self.create_z_kdtrees()	

	def evaluate_z_distances_for_batch(self, latent_z):

		latent_z_sets = {}
		latent_z_sets['robot'] = latent_z[:,:,:int(self.latent_z_dimensionality/2)].detach().cpu().numpy()
		latent_z_sets['env'] = latent_z[:,:,int(self.latent_z_dimensionality/2):].detach().cpu().numpy()
		
		# Robot nearest neighbors.. 
		# Number nearest neighbor
		number_nearest_neighbors = 5
		robot_nn_distances, robot_nn_indices = self.stream_latent_z_dict['robot'].query(latent_z_sets['robot'], k=number_nearest_neighbors)
		env_nn_distances, env_nn_indices = self.stream_latent_z_dict['env'].query(latent_z_sets['env'], k=number_nearest_neighbors)

		# print("Robot Latent Space Average Distance: ", robot_nn_distances.mean())
		return robot_nn_distances, env_nn_distances

	def evaluate_forward_inverse_models(self):

		# Paradigm for forward model.
		# (0) For number of evaluate trajectories:
		# (1) 	Get trajectory from dataset.
		# (1a) 	Parse trajectory into robot and env trajectory T_r, T_e. 
		# (2) 	Encode trajectory into z_R, z_E via global / joint encoder. 
		# (3)	Find nearest neighbor of z_R / z_E in {Z}_R / {Z}_E i.e. z_R^* \ z_E^*.
		# (4) 	Find corresponding z_E^* / z_R^*.
		# (5)	Evaluate |z_E^* - z_E|_2 / .
		# (6) 	Decode z_E to T_e^* / .
		# (7) 	Evaluate |T_e^* - T_e|_2 / . 

		# Remember, steps 0-3 are executed in sample_trajectories_for_evaluating spaces. 

		pass

	def create_evaluate_dynamics_models(self):

		# After we've created latent sets.
		self.define_forward_inverse_models()

		# Eval
		self.evaluate_forward_inverse_models()

class PolicyManager_BatchPretrain(PolicyManager_Pretrain):

	def __init__(self, number_policies=4, dataset=None, args=None):
		super(PolicyManager_BatchPretrain, self).__init__(number_policies, dataset, args)
		self.blah = 0

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((self.args.batch_size,1,self.output_size)),sample_action_seq],axis=1)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((self.args.batch_size,1,self.output_size))],axis=1)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)
		
	def get_batch_element(self, i):

		# Make data_element a list of dictionaries. 
		data_element = []
						
		# for b in range(min(self.args.batch_size, len(self.index_list) - i)):
		# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
		for b in range(self.args.batch_size):

			# print("Index that the get_batch_element is using: b:",b," i+b: ",i+b, self.index_list[i+b])
			# Because of the new creation of index_list in random shuffling, this should be safe to index dataset with.

			# print("Getting data element, b: ", b, "i+b ", i+b, "index_list[i+b]: ", self.index_list[i+b])
			index = self.index_list[i+b]

			if self.args.train:
				self.coverage[index] += 1
			data_element.append(self.dataset[index])

		return data_element

	def get_trajectory_segment(self, i):
	
		if self.args.data in ['ContinuousNonZero','DirContNonZero','ToyContext','DeterGoal']:
			
			# Sample trajectory segment from dataset. 
			sample_traj, sample_action_seq = self.dataset[i:i+self.args.batch_size]
			
			# print("Getting data points from: ",i, " to: ", i+self.args.batch_size)			

			# Subsample trajectory segment. 		
			start_timepoint = np.random.randint(0,self.args.traj_length-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[:, start_timepoint:end_timepoint]	
			sample_action_seq = sample_action_seq[:, start_timepoint:end_timepoint-1]

			self.current_traj_len = self.traj_length

			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

			return concatenated_traj.transpose((1,0,2)), sample_action_seq.transpose((1,0,2)), sample_traj.transpose((1,0,2))
				
		elif self.args.data in global_dataset_list:

			if self.args.data in ['MIME','OldMIME'] or self.args.data=='Mocap':
				# data_element = self.dataset[i:i+self.args.batch_size]
				data_element = self.dataset[self.index_list[i:i+self.args.batch_size]]				
			else:
				data_element = self.get_batch_element(i)

			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length            
			
			batch_trajectory = np.zeros((self.args.batch_size, self.current_traj_len, self.state_size))
			self.subsampled_relative_object_state = np.zeros((self.args.batch_size, self.current_traj_len, self.args.env_state_size))

			# POTENTIAL:
			# for x in range(min(self.args.batch_size, len(self.index_list) - 1)):

			# Changing this selection, assuming that the index_list is corrected to only have within dataset length indices.
			for x in range(self.args.batch_size):
			

				# Select the trajectory for each instance in the batch. 
				if self.args.ee_trajectories:
					traj = data_element[x]['endeffector_trajectory']
				else:
					traj = data_element[x]['demo']

				# Pick start and end.               

				# Sample random start point.
				if traj.shape[0]>self.current_traj_len:

					bias_length = int(self.args.pretrain_bias_sampling*traj.shape[0])

					# Probability with which to sample biased segment: 
					sample_biased_segment = np.random.binomial(1,p=self.args.pretrain_bias_sampling_prob)

					# If we want to bias sampling of trajectory segments towards the middle of the trajectory, to increase proportion of trajectory segments
					# that are performing motions apart from reaching and returning. 

					# Sample a biased segment if trajectory length is sufficient, and based on probability of sampling.
					if ((traj.shape[0]-2*bias_length)>self.current_traj_len) and sample_biased_segment:		
						start_timepoint = np.random.randint(bias_length, traj.shape[0] - self.current_traj_len - bias_length)
					else:
						start_timepoint = np.random.randint(0,traj.shape[0]-self.current_traj_len)

					end_timepoint = start_timepoint + self.current_traj_len


					if self.args.ee_trajectories:
						batch_trajectory[x] = data_element[x]['endeffector_trajectory'][start_timepoint:end_timepoint]
					else:
						batch_trajectory[x] = data_element[x]['demo'][start_timepoint:end_timepoint]
					
					if not(self.args.gripper):
						if self.args.ee_trajectories:
							batch_trajectory[x] = data_element['endeffector_trajectory'][start_timepoint:end_timepoint,:-1]
						else:
							batch_trajectory[x] = data_element['demo'][start_timepoint:end_timepoint,:-1]

					if self.args.data in ['RealWorldRigid', 'RealWorldRigidJEEF']:

						# Truncate the images to start and end timepoint. 
						data_element[x]['subsampled_images'] = data_element[x]['images'][start_timepoint:end_timepoint]

					if self.args.data in ['RealWorldRigidJEEF']:
						self.subsampled_relative_object_state[x] = data_element[x]['relative-object-state'][start_timepoint:end_timepoint]

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				batch_trajectory = (batch_trajectory-self.norm_sub_value)/self.norm_denom_value
				self.normalized_subsampled_relative_object_state = (self.subsampled_relative_object_state - self.norm_sub_value[-self.args.env_state_size:])/self.norm_denom_value[-self.args.env_state_size:]

			# Compute actions.
			action_sequence = np.diff(batch_trajectory,axis=1)
			self.relative_object_state_actions = np.diff(self.normalized_subsampled_relative_object_state, axis=1)

			# Concatenate
			concatenated_traj = self.concat_state_action(batch_trajectory, action_sequence)

			# Scaling action sequence by some factor.             
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			# return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2))
			return concatenated_traj.transpose((1,0,2)), scaled_action_sequence.transpose((1,0,2)), batch_trajectory.transpose((1,0,2)), data_element

	def relabel_relative_object_state_actions(self, padded_action_seq):

		# Here, remove the actions computed from the absolute object states; 
		# Instead relabel the actions in these dimensions into actions computed from the relative state to EEF.. 

		relabelled_action_sequence = padded_action_seq
		# Relabel the action size computes.. 
		# relabelled_action_sequence[..., self.args.robot_state_size:] = self.relative_object_state_actions
		relabelled_action_sequence[..., -self.args.env_state_size:] = self.relative_object_state_actions

		return relabelled_action_sequence

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		# Now assemble inputs for subpolicy.
		
		# Create subpolicy inputs tensor. 			
		subpolicy_inputs = torch.zeros((input_trajectory.shape[0], self.args.batch_size, self.input_size+self.latent_z_dimensionality)).to(device)

		# Mask input trajectory according to subpolicy dropout. 
		self.subpolicy_input_dropout_layer = torch.nn.Dropout(self.args.subpolicy_input_dropout)

		torch_input_trajectory = torch.tensor(input_trajectory).view(input_trajectory.shape[0],self.args.batch_size,self.input_size).to(device).float()
		masked_input_trajectory = self.subpolicy_input_dropout_layer(torch_input_trajectory)

		# Now copy over trajectory. 
		# subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()         
		subpolicy_inputs[:,:,:self.input_size] = masked_input_trajectory

		# Now copy over latent z's. 
		subpolicy_inputs[range(input_trajectory.shape[0]),:,self.input_size:] = latent_z_indices

		# # Concatenated action sequence for policy network's forward / logprobabilities function. 
		# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
		# View time first and batch second for downstream LSTM.
		padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.args.batch_size,self.output_size))],axis=0)

		if self.args.data in ['RealRobotRigidJEEF']:
			padded_action_seq = self.relabel_relative_object_state_actions(padded_action_seq)

		return None, subpolicy_inputs, padded_action_seq

	def construct_dummy_latents(self, latent_z):

		if not(self.args.discrete_z):
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.args.batch_size, self.current_traj_len)).to(device).float()
		latent_b[:,0] = 1.

		return latent_z_indices, latent_b	
		# return latent_z_indices

	def setup_vectorized_environments(self):

		# Don't try to recreate env here, just use the env from the visualizer. 
		self.base_env = self.visualizer.env
		# self.vectorized_environments = gym.vector.SyncVectorEnv([ lambda: GymWrapper(robosuite.make("Door", robots=['Sawyer'], has_renderer=False)) for k in range(self.args.batch_size)])

		if not(isinstance(self.base_env, gym.Env)):
			self.base_env = GymWrapper(self.base_env)
		
		# Vectorized.. 		
		self.vectorized_environment = gym.vector.SyncVectorEnv([ lambda: self.base_env for k in range(self.args.batch_size)])

	def batch_compute_next_state(self, current_state, action):

		# Reset. 
		# Set state.
		# Preprocess action. 
		# Step. 
		# Return state. 

		if self.args.viz_sim_rollout:
			
			####################
			# (0) Reset envs. 
			####################

			self.vectorized_environment.reset()

			####################
			# (1) Set state. 		
			####################

			# Option 1 - do this iteratively - not ideal, but probably fine because this is not the bottleneck. 
			# Option 2 - set using set_attr? - testing this out doesn't seem to work? Anyhow, set_attr iterates internally, so might as well do this ourselves. 
			
			# for k in range(self.args.batch_size):
			for k, environment in enumerate(self.vectorized_environment.envs):
				self.visualizer.set_joint_pose(current_state, env=environment)
			
			####################
			# (2) Preprocess action.
			####################

			action_to_execute = self.preprocess_action(action)

			####################
			# (3) Step
			####################
		
			for k in range(self.args.sim_viz_step_repetition):
				# Use environment to take step.
				env_next_state_dict, _, _, _ = self.visualizer.environment.step(action_to_execute)
				gripper_state = env_next_state_dict[self.visualizer.gripper_key]
				if self.visualizer.new_robosuite:
					joint_state = self.visualizer.environment.sim.get_state()[1][:7]
				else:
					joint_state = env_next_state_dict['joint_pos']

		####################
		# (4) Return State
		####################
			
		else:
			next_state = current_state + action

		return next_state

	def differentiable_rollout(self, trajectory_start, latent_z, rollout_length=None):

		subpolicy_inputs = torch.zeros((self.args.batch_size,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[:,:self.state_dim] = torch.tensor(trajectory_start).to(device).float()
		# subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()
		subpolicy_inputs[:,2*self.state_dim:] = latent_z[0]

		if self.args.batch_size>1:
			subpolicy_inputs = subpolicy_inputs.unsqueeze(0)

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = self.rollout_timesteps-1

		for t in range(length):

			# Get actions from the policy.
			actions = self.policy_network.reparameterized_get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor
			
			# Compute next state. 
			new_state = self.batch_compute_next_state(subpolicy_inputs[t,...,:self.state_dim], action_to_execute)
			# new_state = subpolicy_inputs[t,...,:self.state_dim]+action_to_execute

			# Create new input row. 
			input_row = torch.zeros((self.args.batch_size, 2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
			input_row[:,:self.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[:,self.state_dim:2*self.state_dim] = actions[-1].squeeze(1)
			input_row[:,2*self.state_dim:] = latent_z[t+1]

			# Now that we have assembled the new input row, concatenate it along temporal dimension with previous inputs. 
			if self.args.batch_size>1:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row.unsqueeze(0)],dim=0)
			else:
				subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[...,:self.state_dim].detach().cpu().numpy()
		differentiable_trajectory = subpolicy_inputs[...,:self.state_dim]
		differentiable_action_seq = subpolicy_inputs[...,self.state_dim:2*self.state_dim]
		differentiable_state_action_seq = subpolicy_inputs[...,:2*self.state_dim]

		# For differentiabiity, return tuple of trajectory, actions, state actions, and subpolicy_inputs. 
		return differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs
	
	def batched_visualize_robot_data(self, load_sets=False, number_of_trajectories_to_visualize=None):

		#####################
		# Set number of trajectories to visualize.			
		#####################

		if number_of_trajectories_to_visualize is not None:
			self.N = number_of_trajectories_to_visualize
		else:
			self.N = 400
			# self.N = 100	
			
		#####################
		# Set visualizer based on data / domain. 
		#####################

		self.set_visualizer_object()
		np.random.seed(seed=self.args.seed)

		#####################################################
		# Get latent z sets.
		#####################################################
		
		if not(load_sets):

			#####################################################
			# Select Z indices if necessary.
			#####################################################

			if self.args.split_stream_encoder:
				if self.args.embedding_visualization_stream == 'robot':
					stream_z_indices = np.arange(0,int(self.args.z_dimensions/2))
				elif self.args.embedding_visualization_stream == 'env':
					stream_z_indices = np.arange(int(self.args.z_dimensions/2),self.args.z_dimensions)
				else:
					stream_z_indices = np.arange(0,self.args.z_dimensions)	
			else:
				stream_z_indices = np.arange(0,self.args.z_dimensions)

			#####################################################
			# Initialize variables.
			#####################################################

			# self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
			self.latent_z_set = np.zeros((self.N,len(stream_z_indices)))		
			# These are lists because they're variable length individually.
			self.indices = []
			self.trajectory_set = []
			self.trajectory_rollout_set = []		
			self.rollout_gif_list = []
			self.gt_gif_list = []

			#####################################################
			# Create folder for gifs.
			#####################################################

			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
			# Create save directory:
			upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

			if not(os.path.isdir(upper_dir_name)):
				os.mkdir(upper_dir_name)

			self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
			if not(os.path.isdir(self.dir_name)):
				os.mkdir(self.dir_name)

			self.max_len = 0

			#####################################################
			# Initialize variables.
			#####################################################

			self.shuffle(len(self.dataset)-self.test_set_size, shuffle=True)
		
			#############################
			# For appropriate number of batches: 
			#############################

			for j in range(self.N//self.args.batch_size):
			
				#############################		
				# (1) Encode trajectory. 
				#############################

				if self.args.setting in ['learntsub','joint', 'queryjoint']:
					print("Embed in viz robot data")
					
					input_dict, var_dict, eval_dict = self.run_iteration(0, j, return_dicts=True, train=False)
					latent_z = var_dict['latent_z_indices']
					sample_trajs = input_dict['sample_traj']
				else:
					print("Running iteration of segment in viz")
					latent_z, sample_trajs, _, data_element = self.run_iteration(0, j, return_z=True, and_train=False)

				#############################
				# (2) 
				#############################

				# Create env for batch.
				self.per_batch_env_management(data_element[0])

				#############################
				# (3) Rollout for each trajectory in batch.
				#############################

				trajectory_rollout, _, _, _ = self.differentiable_rollout(sample_trajs[0], latent_z)

				# Need to add some stuff here to mimic get_robot_visuals' list management.

				if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
					unnorm_gt_trajectory = (sample_trajs*self.norm_denom_value)+self.norm_sub_value
					unnorm_pred_trajectory = (trajectory_rollout*self.norm_denom_value) + self.norm_sub_value
				else:
					unnorm_gt_trajectory = sample_trajs
					unnorm_pred_trajectory = trajectory_rollout

				#############################
				# (4) Visualize for every trajectory in batch. 
				#############################

				for b in range(self.args.batch_size):

					# First visualize ground truth gif. 					
					self.ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GIF_GT.gif".format(i), return_and_save=True, end_effector=self.args.ee_trajectories, task_id=env_name)
					
					# Now visualize rollout. 
					

					# Copy it to global lists.
					self.gt_gif_list.append(copy.deepcopy(self.ground_truth_gif))
					self.rollout_gif_list.append(copy.deepcopy(self.rollout_gif))


			# Get MIME embedding for rollout and GT trajectories, with same Z embedding. 
			embedded_z = self.get_robot_embedding()

		# Save animations.
		gt_animation_object = self.visualize_robot_embedding(embedded_z, gt=True)
		rollout_animation_object = self.visualize_robot_embedding(embedded_z, gt=False)

		# Save webpage
		self.write_results_HTML()
		# Save webpage plots
		self.write_results_HTML('Plot')

		self.write_embedding_HTML(gt_animation_object,prefix="GT")
		self.write_embedding_HTML(rollout_animation_object,prefix="Rollout")
#####################################################################################################################################################################################################################
