import load_imagenet
from keras.applications.vgg16 import VGG16
import numpy as np
from keras.applications.imagenet_utils import decode_predictions

vgg = VGG16(weights='imagenet', include_top=True)
# Evaluate keras pretrained
train, test = load_imagenet.get_imagenet_data()
classes = train.keys()
correct = 0
total = 0
for k in classes:
	print("predicting class k")
	predicts = vgg.predict(train[k][0])
	decode = decode_predictions(predicts)
	top = [int(p[0][0].strip('n'))==k for p in decode]
	correct += sum(top)
	total += len(top)
print(correct/total)