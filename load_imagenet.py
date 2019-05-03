# imagenet dataloader
import os
import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np

path_to_imagenet = './dataset/tiny-imagenet-200/'
m = 15
path_to_activations = './'

def get_imagenet_data():
	# find all the classes
	classes = os.listdir(os.path.join(path_to_imagenet, 'train'))
	# select subset of m classes
	to_use = list(set([random.choice(classes) for _ in range(2*m)]))[:m]
	# write to txt file in same area as activations
	with open(os.path.join(path_to_activations, 'used_classes.txt'), 'w') as f:
		print("hi")
		f.write('\n'.join(to_use))

	# load all images for those classes by class
	train_by_class = {}
	test_by_class = None
	for cls in to_use:
		print(cls)
		images_in_class = []
		img_path = os.path.join(path_to_imagenet, 'train/%s/images/' % cls)
		for img in [f for f in os.listdir(img_path) if '.JPEG' in f]:
			im = resize(imread(os.path.join(img_path, img)), (224,224,3))
			images_in_class.append(im)
		train_by_class[cls] = (np.stack(tuple(images_in_class)), [int(cls[1:]) for _ in range(len(images_in_class))])
	# train_by_class[cls] = array

	return train_by_class, test_by_class
