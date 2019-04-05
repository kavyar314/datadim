import numpy as np
import json

path_to_activations = './data/vgg/cifar10_train_c%d.npy'
out_file = 'centers.npy'
out_file_vectors = 'vectors_between_centers.npy'

class_list = range(10)


def find_centers(class_number):
	file_class = path_to_activations % class_number
	points_by_layer = np.load(file_class).item()
	centers = {}
	for layer, h in points_by_layer.items():
		centers[layer] = np.mean(h, axis=0).flatten()
	return centers

def centers_by_layer():
	centers_by_class = {}
	for class_number in class_list:
		print("class_number", class_number)
		centers_by_class[class_number] = find_centers(class_number)
	centers_by_layer_dict = {}
	print("fixing dictionary")
	for k in centers_by_class[0].keys():
		centers_by_layer_dict[k] = {}
		for i in class_list:
			centers_by_layer_dict[k][i] = centers_by_class[i][k]
	return centers_by_layer_dict

def pairwise_vectors(centers_by_layer_dictionary):
	d = centers_by_layer_dictionary
	out_dict = {}
	for layer in centers_by_layer_dictionary.keys():
		for i in class_list:
			for j in class_list[i+1:]:
				vect = d[layer][i] - d[layer][j]
				out_dict[(i, j)] = vect

	return out_dict


if __name__ == '__main__':
	d = centers_by_layer()
	np.save(out_file, d)
	vectors = pairwise_vectors(d)
	np.save(out_file_vectors, vectors)

