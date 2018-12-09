from __future__ import print_function, division

import argparse
from collections import defaultdict
from glob import glob
import logging
import os
import sys

import keras
from keras.datasets import cifar10
import keract
import numpy as np

from IPython import embed

from cifar10vgg import cifar10vgg

MAX_PER_CLASS = 1000
CIFAR10_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
OUT_PATH = './'

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


def svd(args):
    '''
    NB this can't be run without fixing file paths
    '''
    filepaths = glob("data/cifar10_{}*.npy".format(args.split))
    for filepath in filepaths:
        h_by_layer = np.load(filepath).item()

        for layer, h in h_by_layer.items():
            logging.info("Computing SVD for %s layer %s activations of shape %s", filepath, layer, h.shape)
            # TODO: Should we be taking the multilinear SVD, or reshaping h to (1000, .)?
            # KR: I think we should flatten first b/c in the last couple layers (FC), we will already have "flat" matrices.
            # KR: We might be able to speed up by not calculating u, v if we aren't planning to use them
            u, s, vh = np.linalg.svd(h.reshape(MAX_PER_CLASS, -1), full_matrices=False)
            # Take the multilinear SVD, and flatten singular values
            # u, s, vh = np.linalg.svd(h, full_matrices=False)
            # s = s.flatten()

            path_full = OUT_PATH + 'singular_values'
            if 'singular_values' not in os.listdir(OUT_PATH):
                os.makedirs(path_full)
            savefile = '{}/singularValues_{}_{}.npy'.format(path_full, filepath.strip('data/').strip('.npy'), layer.split('/')[0])
            np.save(savefile, s)
            # TODO: group s and then store

def pairwise_svd(args):
    files = glob("data/cifar10_{}*.npy".format(args.split))
    for file1 in files:
        for file2 in files:
            h1_by_layer = np.load(file1).item()
            h2_by_layer = np.load(file2).item()
            for layer in h1_by_layer.keys():
                h_total = np.vstack(h1_by_layer[layer], h2_by_layer[layer])
                dim = h1_by_layer[layer].shape[0] + h2_by_layer[layer].shape[0]
                _, s, _ = np.linalg.svd(h_total.reshape(dim, -1), full_matrices=False)
                path_full = OUT_PATH + 'pairwise_sv'
                if 'pairwise_sv' not in os.listdir(OUT_PATH):
                    os.makedirs(path_full)
                savefile = '{}/singularValues_{}_{}_{}.npy'.format(path_full, file1.strip('data/').strip('.npy'), file2.split('_')[-1].strip('.npy'), layer.split('/')[0])
                np.save(savefile, s)


    

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

