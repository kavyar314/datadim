import numpy as np
import os
import datetime
import time

import argparse

OUT_FILE = './singular_value_stats.csv'

PATH_TO_SV = './singular_values/'

FAILED_FILES = './failed_files.txt'

n_stats = 'all'

p_list = [5, 10, 20, 30, 50, 75, 100]
model = 'vgg'

STAT_FILE_FORMAT = './singular_value_statistics_%s.csv'

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
	mean = np.mean(singular_values/singular_values[0])
	stdev = np.std(singular_values/singular_values[0])
	out = "{}, {}, {}, {}, {}, {} \n".format(details[1], details[2], details[3], details[5], mean, stdev)
	with open(OUT_FILE, 'a') as f:
		f.write(out)

def all_the_stats(singular_values, specs, outfile_path):
	#stuff
	# 5%, high p-norm of spectrum, \sigma_2/\sigma_1, \sigma_3/\sigma_1
	details = specs.split('_')
	five_percent = singular_values[singular_values.shape[0]/20]/singular_values[0]
	p_norms = [np.linalg.norm(singular_values/singular_values[0], p) for p in p_list]
	first_dropoff = singular_values[1]/singular_values[0]
	second_dropoff = singular_values[2]/singular_values[0]
	out = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(details[1], details[2], details[3], details[5], five_percent, ','.join(p_norms), first_dropoff, second_dropoff)
	with open(outfile_path, 'a') as f:
		f.write(out)


if __name__ == '__main__':
	#parser = argparse.ArgumentParser()
	#parser.add_argument('--n_stats', type=str)
	#args = parser.parse_args()
	full_path_to_sv = os.path.join(PATH_TO_SV, model)
	sv_files = [f for f in os.listdir(full_path_to_sv) if '.npy' in f]
	outfile_path = STAT_FILE_FORMAT % datetime.datetime.fromtimestamp(time.time()).isoformat()
	for f in sv_files:
		try:
			if n_stats=='all':
				all_the_stats(np.load(os.path.join(full_path_to_sv, f)), f.strip('.npy'), outfile_path)
			else:
				calc_stats(np.load(os.path.join(full_path_to_sv, f)), f.strip('.npy'))
		except:
			with open(FAILED_FILES, 'a') as fail:
				fail.write(f)
			print("{} failed".format(f))