from matplotlib import pyplot as plt
import numpy as np
import logging
import os

PATH_TO_SV = './singular_values/'

OUT_PATH = './singular_values/plots/'
if OUT_PATH not in os.listdir('./'):
	os.makedirs(OUT_PATH)

def plot_singular_values(sv_array, specs):
	if sv_array[0] != 1:
		sv_array/sv_array[0]
	fname = specs + '.pdf'
	plt.figure()
	plt.plot(sv_array)
	plt.savefig(os.path.join(OUT_PATH, fname))
	plt.close()

def open_and_process_singular_values(fname):
	singular_vals = np.load(fname)
	return singular_vals/singular_vals[0]

if __name__ == '__main__':
	sv_files = os.listdir(PATH_TO_SV)
	for f in sv_files:
		logging.info('Making plot for %s', f)
		sv = open_and_process_singular_values(f)
		specs = f.strip('.npy')
		plot_singular_values(sv, specs)


