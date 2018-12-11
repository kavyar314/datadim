import numpy as np
import os

from compute_stats import all_the_stats

path_to_pairwise = '/mnt/datadim/datadim/pairwise_sv'
outfile_path = './pairwise_statistics.csv'

if __name__ == '__main__':
	sv_files = [f for f in os.listdir(path_to_pairwise) if '.npy' in f]
	for f in sv_files:
		sv = np.load(os.path.join(path_to_pairwise, f))
		spec = f.split('_')
		spec.pop()
		spec = ''.join(spec).strip('.npy')
		print spec
		all_the_stats(sv, spec, outfile_path)
