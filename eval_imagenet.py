import load_imagenet
from keras.applications.vgg16 import VGG16
import numpy as np
from keras.applications.imagenet_utils import decode_predictions

top_five = True

vgg = VGG16(weights='imagenet', include_top=True)
# Evaluate keras pretrained
train, test = load_imagenet.get_imagenet_data()
classes = train.keys()
correct = 0
total = 0
for k in classes:
	print("predicting class", k)
	predicts = vgg.predict(train[k][0])
	decode = decode_predictions(predicts)
	check_equal = lambda key, options: [sum([int(p[i][0].strip('n'))==key for i in range(len(p))]) > 0 for p in options]
	if top_five:
		top = check_equal(k, decode)
	else:
		top = [int(p[0][0].strip('n'))==k for p in decode]
	correct += sum(top)
	total += len(top)
	print(correct, total, correct/total)
print(correct/total)