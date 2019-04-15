import numpy as np
import os

from compute_stats import all_the_stats

path_to_pairwise = './pairwise_sv_centered/vgg'
outfile_path = './pairwise_statistics_centered.csv'

if __name__ == '__main__':
	sv_files = [f for f in os.listdir(path_to_pairwise) if '.npy' in f and 'singularValues' in f]
	for f in sv_files:
		sv = np.load(os.path.join(path_to_pairwise, f))
		spec = f.split('_')
		spec.pop(0)
		spec = '_'.join(spec).strip('.npy')
		print(spec)
		all_the_stats(sv, spec, outfile_path)
