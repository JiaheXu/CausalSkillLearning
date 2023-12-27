import torch
import numpy as np
import glob, os, sys, argparse
import rosbag
import rospy
import pdb
import sklearn.manifold as skl_manifold
from sklearn.decomposition import PCA
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import cv2


file_path = "/home/mmpug/CausalSkillLearning/TrainingLogs/MAGI_1224/LatentSetDirectory/E100000_C100000"


latent_z_set = np.load(os.path.join(file_path, "LatentSet.npy"))
gt_trajectory_set = np.load(os.path.join(file_path, "GT_TrajSet.npy"), allow_pickle=True)
embedded_zs = np.load(os.path.join(file_path, "EmbeddedZSet.npy"), allow_pickle=True)
task_id_set = np.load(os.path.join(file_path, "TaskIDSet.npy"), allow_pickle=True)



N = 500

print("latent_z_set: ", latent_z_set.shape)
print("gt_trajectory_set: ", gt_trajectory_set.shape)
print("embedded_zs: ", embedded_zs.shape)
print("task_id_set: ", task_id_set.shape)

def get_robot_embedding(return_tsne_object=False, perplexity=None): #!!! here
	# # Mean and variance normalize z.
	# mean = self.latent_z_set.mean(axis=0)
	# std = self.latent_z_set.std(axis=0)
	# normed_z = (self.latent_z_set-mean)/std
	normed_z = latent_z_set #!!! here

	if perplexity is None:
		perplexity = 5
		
	print("Perplexity: ", perplexity)

	tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=perplexity)
	embedded_zs = tsne.fit_transform(normed_z)

	scale_factor = 1
	scaled_embedded_zs = scale_factor*embedded_zs

	if return_tsne_object:
		return scaled_embedded_zs, tsne
	else:
		return scaled_embedded_zs

def plot_embedding(embedded_zs, title, shared=False, trajectory=False): #!!! here
	
	fig = plt.figure()
	ax = fig.gca()
		
	if shared:
		colors = 0.2*np.ones((2*N))
		colors[N:] = 0.8
	else:
		colors = 0.2*np.ones((N))

	ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet')
		
	# Title. 
	ax.set_title("{0}".format(title),fontdict={'fontsize':15})
	fig.canvas.draw()

	fig.savefig( title[ -3:-1] + ".png" )

	return None


embedded_z_dict = {}
embedded_z_dict['perp5'] = get_robot_embedding(perplexity=5) #!!! here
embedded_z_dict['perp10'] = get_robot_embedding(perplexity=10)
embedded_z_dict['perp30'] = get_robot_embedding(perplexity=30)

# # Now plot the embedding.

stat_dictionary = {}
stat_dictionary['counter'] = 0
stat_dictionary['i'] = 0
stat_dictionary['epoch'] = 100000
stat_dictionary['batch_size'] = 64
statistics_line = "Epoch: {0}, Count: {1}, I: {2}, Batch: {3}".format(stat_dictionary['epoch'], stat_dictionary['counter'], stat_dictionary['i'], stat_dictionary['batch_size'])

image_perp5 = plot_embedding(embedded_z_dict['perp5'], title="Z Space {0} Perp_05".format(statistics_line)) #!!! here
image_perp10 = plot_embedding(embedded_z_dict['perp10'], title="Z Space {0} Perp_10".format(statistics_line))
image_perp30 = plot_embedding(embedded_z_dict['perp30'], title="Z Space {0} Perp_30".format(statistics_line))

# cv2.imwrite( "perp5.png", image_perp5)
# cv2.imwrite( "perp10.png", image_perp10)
# cv2.imwrite( "perp30.png", image_perp30)
			