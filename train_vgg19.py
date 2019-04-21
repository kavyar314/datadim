# training vgg 19

import models.general_vgg

m = general_vgg.cifar10vgg(4, train=True, weight_file='cifar10_vgg19weights.h5')