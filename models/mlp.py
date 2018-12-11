# From https://github.com/geifmany/cifar-vgg
# Licenced under GPLv3

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

class MLP:
    def __init__(self, train=True, num_layers=14, hidden_dim=1000, weight_file='mlp.h5'):
        self.num_classes = 10
        #self.weight_decay = 0.0005
        self.x_shape = [32,32,3]
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights(weight_file)


    def build_model(self):
        model = Sequential()
        #weight_decay = self.weight_decay

        model.add(Flatten(input_shape=self.x_shape))

        for l in range(self.num_layers):
            model.add(Dense(self.hidden_dim))
            model.add(Activation('relu'))

        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepochs = 450
        learning_rate = 0.01
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        print("x_train:", x_train.shape)
        print("y_train:", y_train.shape)
        print("model:", model.summary())

        checkpoint_filepath = 'weights/mlp_l{}_h{}'.format(self.num_layers, self.hidden_dim) + "_e{epoch:02d}_vl{val_loss:.4f}.h5"
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

        dataflow = datagen.flow(x_train, y_train,
                                batch_size=batch_size)

        #historytemp = model.fit(x_train, y_train, epochs=maxepochs, steps_per_epoch=x_train.shape[0] // batch_size)
        historytemp = model.fit_generator(dataflow,
                                          steps_per_epoch=x_train.shape[0] // batch_size,
                                          epochs=maxepochs,
                                          validation_data=(x_test, y_test),
                                          callbacks=[reduce_lr, checkpoint],
                                          workers=6,
                                          use_multiprocessing=True,
                                          verbose=2)
        model.save_weights('weights/mlp_l{}_h{}.h5'.format(self.num_layers, self.hidden_dim))

        return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = MLP(train=True, num_layers=12, hidden_dim=1000)

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)

