import numpy as np
import os

CENTERS = True

# cifar 10 with VGG-16
class_list = range(10)
layer_list = range(1,15)
if not CENTERS:
	out_file = 'singular_vector_distances.csv'
else:
	out_file = 'singular_vector_center_vector_distances.csv'
path_to_svec_one = './singular_values_vecs/vgg/singularVectors_cifar10_train_c%d_%s_%d.npy'
path_to_svec_pair = './pairwise_sv/vgg/'
fmt = 'singularVectors_cifar10_train_c%d_c%d_activation_%d.npy'
center_vec_dict = np.load('vectors_between_centers.npy').item()

def cosine_distance(v1, v2):
	return np.dot(v1.T, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

if __name__ == '__main__':
	csv_out = "%d,%d,%d,%04f,%04f,%04f,%04f\n" # layer, class 1, class 2, (see notebook for order)
	files = [svec_file for svec_file in os.listdir(path_to_svec_pair) if 'Vector' in svec_file]
	for f in files:
		print(f)
		attr = f.split('_')
		c1 = int(attr[3][1])
		c2 = int(attr[4][1])
		layer = int(attr[-1].strip('.npy'))
		act_inp = attr[-2]
		vecs_c1 = np.load(path_to_svec_one % (c1, act_inp,layer))
		print("vecs_c1 shape is", vecs_c1.shape)
		# get the first singular vector for c1
		u_1_1 = vecs_c1.T[0,:]
		# load individual singular vector for c2
		vecs_c2 = np.load(path_to_svec_one % (c2, act_inp, layer))
		print("vecs_c2 shape is", vecs_c2.shape)
		# get the first singular vector for c2
		u_2_1 = vecs_c2.T[0,:]
		# load pair singular vector for c1, c2
		vecs_pair = np.load(os.path.join(path_to_svec_pair,f))
		print("vecs_pair shape is", vecs_pair.shape)
		# get first and second vectors
		u_12_1 = vecs_pair.T[0, :]
		u_12_2 = vecs_pair.T[1, :]
		# find cosine distances
		if not CENTERS:
			c1_11 = cosine_distance(u_1_1, u_12_1)
			c1_12 = cosine_distance(u_1_1, u_12_2)
			c2_11 = cosine_distance(u_2_1, u_12_1)
			c2_12 = cosine_distance(u_2_1, u_12_2)
			next_line = csv_out % (layer, c1, c2, c1_11, c1_12, c2_11, c2_12)
		if CENTERS:
			center_vec = center_vec_dict[(c1,c2)]
			c1_1_c = cosine_distance(u_1_1, center_vec)
			c2_1_c = cosine_distance(u_2_1, center_vec)
			c12_1_c = cosine_distance(u_12_1, center_vec)
			c12_2_c = cosine_distance(u_12_2, center_vec)
			next_line = csv_out % (layer, c1, c2, c1_1_c, c2_1_c, c12_1_c, c12_2_c)
		with open(out_file, 'a') as out:
			out.write(next_line)
