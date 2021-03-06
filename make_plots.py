import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import logging
import os

PATH_TO_SV = './singular_values/'

OUT_PATH = './singular_values/plots/on_same/'

FAILED_FILES = './plot_failed_files.txt'

LOG = True

if 'plots' not in os.listdir('./singular_values'):
	os.makedirs(OUT_PATH)

if 'on_same' not in os.listdir('./singular_values/plots'):
	os.makedirs(OUT_PATH)

def plot_singular_values(sv_arrays, specs, log=LOG):
	'''
	sv_arrays is a list of list of singular values, ordered by class (0-9)
	need to 
	'''
	for sv_array in sv_arrays:
		if sv_array[0] != 1:
			sv_array/sv_array[0]
		fname = specs + '.pdf'
	if log:
		sv_arrays = [np.log(sv_array) for sv_array in sv_arrays]
		fname = 'log' + fname
	plt.figure()
	for sv_array in sv_arrays:
		plt.plot(sv_array)
	plt.legend(range(len(sv_arrays)))
	plt.savefig(os.path.join(OUT_PATH, fname))
	plt.close()

def open_and_process_singular_values(fname):
	# singularValues_cifar10_train_c3_activation_15.npy
	# [0]_[1]model_[2]split_[3]class_[4]_[5]layer.npy
    #print fname
	singular_vals = np.load(os.path.join(PATH_TO_SV + fname))
	return singular_vals/singular_vals[0]

if __name__ == '__main__':
	sv_files = [f for f in os.listdir(PATH_TO_SV) if '.npy' in f]
	layers = list(set([f.split('_')[5] for f in sv_files]))
        #print layers
	by_layer = dict(zip(layers, [[f for f in sv_files if f.split('_')[5]==l] for l in layers]))
        #print len(by_layer[layers[0]])
        for layer in layers:
		sv_list = []
		list_spec = by_layer[layer][0].strip('.npy').split('_')
		specs = '_'.join(list_spec[:3] + list_spec[4:])
		for f in by_layer[layer]: #this area is broken -- need to only plot after the list is complete and naming needs to be fixed
			logging.info('Making plot for %s', f)
			try:
				sv = open_and_process_singular_values(f)
				sv_list.append(sv)
			except:
				with open(FAILED_FILES, 'a') as fail:
					fail.write(f)
				print('{} failed'.format(f))
		plot_singular_values(sv_list, specs)


