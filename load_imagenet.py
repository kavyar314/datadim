# imagenet dataloader
import os
import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

path_to_imagenet = './dataset/tiny-imagenet-200/'
m = 10
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
		imgs = [f for f in os.listdir(img_path) if '.JPEG' in f]
		used_imgs = random.sample(imgs, 100)
		for img in used_imgs:
			im = image.img_to_array(image.load_img(os.path.join(img_path, img), target_size=(224,224)))
			images_in_class.append(im)
		train_by_class[int(cls[1:])] = (preprocess_input(np.stack(tuple(images_in_class))), [int(cls[1:]) for _ in range(len(images_in_class))])
	# train_by_class[cls] = array

	return train_by_class, test_by_class
