import numpy as np
import os

OUT_FILE = './singular_value_stats.csv'

PATH_TO_SV = './singular_values/'

def calc_mean(singular_values):
	return np.mean(singular_values)

def calc_stdev(singular_values):
	return np.std(singular_values)

def calc_stats(singular_values, specs):
	'''
	calculates mean and stdev and saves to csv
	'''
	# singularValues_cifar10_train_c3_activation_15.npy
	# [0]_[1]model_[2]split_[3]class_[4]_[5]layer.npy
	details = specs.split('_')
	mean = np.mean(singular_values)
	stdev = np.std(singular_values)
	out = "{}, {}, {}, {}, {}, {}".format{details[1], details[2], details[3], details[5], mean, stdev}
	with open(OUT_FILE, 'a') as f:
		f.write(out)

if __name__ == '__main__':
	sv_files = [f for f in os.listdir(PATH_TO_SV) if '.npy' in f]
	for f in sv_files:
		calc_stats(np.load(os.path.join(PATH_TO_SV, f)), f.strip('.npy'))