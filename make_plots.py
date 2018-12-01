import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import logging
import os

PATH_TO_SV = './singular_values/'

OUT_PATH = './singular_values/plots/'

LOG = True

if 'plots' not in os.listdir('./singular_values'):
	os.makedirs(OUT_PATH)

def plot_singular_values(sv_array, specs, log=LOG):
	if sv_array[0] != 1:
		sv_array/sv_array[0]
	fname = specs + '.pdf'
	if log:
		sv_array = np.log(sv_array)
		fname = 'log' + fname
	plt.figure()
	plt.plot(sv_array)
	plt.savefig(os.path.join(OUT_PATH, fname))
	plt.close()

def open_and_process_singular_values(fname):
        print fname
	singular_vals = np.load(os.path.join(PATH_TO_SV + fname))
	return singular_vals/singular_vals[0]

if __name__ == '__main__':
	sv_files = [f for f in os.listdir(PATH_TO_SV) if '.npy' in f]
	for f in sv_files:
		logging.info('Making plot for %s', f)
		sv = open_and_process_singular_values(f)
		specs = f.strip('.npy')
		plot_singular_values(sv, specs)


