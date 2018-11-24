from __future__ import print_function, division

import argparse
from collections import defaultdict
import logging
import os
import sys

import keras
from keras.datasets import cifar10
import keract
import numpy as np

from IPython import embed

from models.cifar10vgg import cifar10vgg

MAX_PER_CLASS = 1000
CIFAR10_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

logging.basicConfig(level=logging.DEBUG)


def make_splits():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    train_by_class = {}
    test_by_class = {}

    for cls in CIFAR10_CLASSES:
        # Randomly sample datapoints from class
        idx, _ = np.where(y_train == cls)
        idx_train = np.random.choice(idx, MAX_PER_CLASS, replace=False)
        train_by_class[cls] = x_train[idx_train]

        idx_test, _ = np.where(y_test == cls)
        test_by_class[cls] = x_test[idx_test]
        assert(idx_test.size <= MAX_PER_CLASS)

        logging.info("Class %d Train: %s Test %s", cls, train_by_class[cls].shape, test_by_class[cls].shape)

    return train_by_class, test_by_class


def infer(args):
    train_by_class, test_by_class = make_splits()
    X_by_class = train_by_class if args.split == "train" else test_by_class

    model = cifar10vgg(train=False, weight_file="data/cifar10vgg.h5")

    for cls, X in X_by_class.items():
        by_layer = defaultdict(list)

        for i in range(0, X.shape[0], args.bs):
            logging.info("Inference for class %d: %d/%d", cls, i, X.shape[0])
            batch = X[i:i+args.bs]
            batch = model.normalize_production(batch)
            activations = keract.get_activations(model.model, batch)

            for layer, h in activations.items():
                # Select activations after Relu and Softmax
                if layer.startswith("activation_"):
                    # TODO(ajayjain): Do we need to know whether this example
                    # was misclassified? The dimensionality of the true positive
                    # examples may be less than the TP + FN examples
                    by_layer[layer].append(h)

        # Concatenate activations and save tensors
        for layer, h_list in by_layer.items():
            by_layer[layer] = np.concatenate(h_list, axis=0)

        os.makedirs("data", exist_ok=True)
        savefile = "data/cifar10_{}_c{}.npy".format(args.split, cls)
        logging.info("Saving activations to %s", savefile)
        np.save(savefile, by_layer)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Launcher for datadim experiments")
    parser.add_argument("task", metavar="T", type=str)
    parser.add_argument("--seed", default=1234, required=False, type=int)
    parser.add_argument("--bs", default=10, required=False, type=int)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.task in globals():
        task_function = globals()[args.task]
        task_function(args)

