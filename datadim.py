from __future__ import print_function, division

import argparse

import keras
from keras.datasets import cifar10
import numpy as np

def make_cifar10_splits():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Launcher for datadim experiments")
    parser.add_argument("task", metavar="T", type=str)
    parser.parse_args()
    args = parser.parse_args()

    if args.task in globals():
        task_function = globals()[args.task]
        task_function()

