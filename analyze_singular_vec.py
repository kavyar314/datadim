import numpy as np
import os

# cifar 10 with VGG-16
class_list = range(10)
layer_list = range(1,15)
out_file = 'singular_vector_distances.csv'
path_to_svec_one = './singular_values_vecs/vgg/singularVectors_cifar10_train_c%d_activation_%d.npy'
path_to_svec_pair = './pairwise_sv/vgg/'
fmt = 'singularVectors_cifar10_train_c%d_c%d_activation_%d.npy'

def cosine_distance(v1, v2):
	return np.dot(v1.T, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

# if __name__ == '__main__':
# 	csv_out = "%d,%d,%d,%04f,%04f,%04f,%04f" # layer, class 1, class 2, (see notebook for order)
# 	for layer in layer_list:
# 		print("layer =", layer)
# 		for i in range(len(class_list)):
# 			c1 = class_list[i]
# 			# load individual singular vector for c1
# 			vecs_c1 = np.load(path_to_svec_one % (c1,layer))
# 			# get the first singular vector for c1
# 			u_1_1 = vecs_c1[0,:]
# 			for c2 in class_list[i+1:]:
# 				# load individual singular vector for c2
# 				vecs_c2 = np.load(path_to_svec_one % (c2, layer))
# 				# get the first singular vector for c2
# 				u_2_1 = vecs_c2[0,:]
# 				# load pair singular vector for c1, c2
# 				vecs_pair = np.load(path_to_svec_pair % (c1, c2, layer))
# 				# get first and second vectors
# 				u_12_1 = vecs_pair[0, :]
# 				u_12_2 = vecs_pair[1, :]
# 				# find cosine distances
# 				c1_11 = cosine_distance(u_1_1, u_12_1)
# 				c1_12 = cosine_distance(u_1_1, u_12_2)
# 				c2_11 = cosine_distance(u_2_1, u_12_1)
# 				c2_12 = cosine_distance(u_2_1, u_12_2)
# 				next_line = csv_out % (layer, c1, c2, c1_11, c1_12, c2_11, c2_12)
# 				with open(out_file, 'a') as out:
# 					out.write(next_line)

if __name__ == '__main__':
	csv_out = "%d,%d,%d,%04f,%04f,%04f,%04f\n" # layer, class 1, class 2, (see notebook for order)
	files = [svec_file for svec_file in os.listdir(path_to_svec_pair) if 'Vector' in svec_file]
	for f in files:
		print(f)
		attr = f.split('_')
		c1 = int(attr[3][1])
		c2 = int(attr[4][1])
		layer = int(attr[-1].strip('.npy'))
		vecs_c1 = np.load(path_to_svec_one % (c1,layer))
		print("vecs_c1 shape is", vecs_c1.shape)
		# get the first singular vector for c1
		u_1_1 = vecs_c1.T[0,:]
		# load individual singular vector for c2
		vecs_c2 = np.load(path_to_svec_one % (c2, layer))
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
		c1_11 = cosine_distance(u_1_1, u_12_1)
		c1_12 = cosine_distance(u_1_1, u_12_2)
		c2_11 = cosine_distance(u_2_1, u_12_1)
		c2_12 = cosine_distance(u_2_1, u_12_2)
		next_line = csv_out % (layer, c1, c2, c1_11, c1_12, c2_11, c2_12)
		with open(out_file, 'a') as out:
			out.write(next_line)
