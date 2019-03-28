import numpy as np

path_to_activations = './data/vgg/cifar10_train_c%d.npy'
out_file = 'centers.json'

def find_centers(class_number):
	file_class = path_to_activations % class_number
	points_by_layer = np.load(file_class).item()
	centers = {}
	for layer, h in points_by_layer.items():
		centers[layer] = np.mean(h, axis=0).flatten()
	return centers

def centers_by_layer():
	centers_by_class = {}
	for class_number in range(10):
		print("class_number", class_number)
		centers_by_class[class_number] = find_centers(class_number)
	centers_by_layer_dict = {}
	print("fixing dictionary")
	for k in centers_by_class[0].keys():
		centers_by_layer_dict[k] = {}
		for i in range(10):
			centers_by_layer_dict[k][i] = centers_by_class[i][k]
	return centers_by_layer_dict

if __name__ == '__main__':
	d = centers_by_layer()
	with open(out_file) as f:
		f.write(d)


