from __future__ import print_function, division

import argparse
from collections import defaultdict
from glob import glob
import logging
import os
import sys

from joblib import Parallel, delayed
import multiprocessing

import keras
from keras.datasets import cifar10
from keras import optimizers
import keract
import numpy as np

from IPython import embed

from models.cifar10vgg import cifar10vgg
from models.mlp import MLP

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
        train_by_class[cls] = (x_train[idx_train], y_train[idx_train])

        idx_test, _ = np.where(y_test == cls)
        test_by_class[cls] = (x_test[idx_test], y_test[idx_test])
        assert(idx_test.size <= MAX_PER_CLASS)

        logging.info("Class %d Train: %s Test %s", cls, train_by_class[cls][0].shape, test_by_class[cls][0].shape)

    return train_by_class, test_by_class


def _create_model(model_name):
    if model_name == "vgg":
        model = cifar10vgg(train=False, weight_file="data/cifar10vgg.h5")
    elif model_name == "mlp5":
        model = MLP(train=False, num_layers=5, hidden_dim=1000, weight_file="models/weights/mlp_l5_h1000.h5")
    elif model_name == "mlp8":
        model = MLP(train=False, num_layers=8, hidden_dim=1000, weight_file="models/weights/mlp_l8_h1000.h5")
    elif model_name == "mlp12":
        model = MLP(train=False, num_layers=12, hidden_dim=1000, weight_file="models/weights/mlp_l12_h1000.h5")

    return model


def evaluate(args):
    assert(args.split == "train")

    train_by_class, test_by_class = make_splits()
    XY_by_class = train_by_class if args.split == "train" else test_by_class

    model = _create_model(args.model)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    accuracies = []
    for cls, (X, y) in XY_by_class.items():
        assert(X.shape[0] == 1000)

        X = model.normalize_production(X)
        y = keras.utils.to_categorical(y, model.num_classes)
        _loss, acc = model.model.evaluate(X, y, batch_size=100, verbose=2)
        accuracies.append(acc)

        print("Train 1000 acc, model {}, class {}: {}".format(args.model, cls, acc))

    print("Train 1000 acc, model {}, overall: {}".format(args.model, np.mean(accuracies)))


def infer(args):
    train_by_class, test_by_class = make_splits()
    XY_by_class = train_by_class if args.split == "train" else test_by_class

    model = _create_model(args.model)

    for cls, (X, _y) in XY_by_class.items():
        by_layer = defaultdict(list)

        for i in range(0, X.shape[0], args.bs):
            logging.info("Inference for class %d: %d/%d", cls, i, X.shape[0])
            batch = X[i:i+args.bs]
            batch = model.normalize_production(batch)
            activations = keract.get_activations(model.model, batch)

            by_layer["input_0"].append(batch)  # Input after normalization

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

        os.makedirs("data/{}".format(args.model), exist_ok=True)
        savefile = "data/{}/cifar10_{}_c{}.npy".format(args.model, args.split, cls)
        logging.info("Saving activations to %s", savefile)
        np.save(savefile, by_layer)


def svd(args):
    '''
    NB this can't be run without fixing file paths
    '''
    filepaths = glob("data/{}/cifar10_{}*.npy".format(args.model, args.split))
    for filepath in filepaths:
        h_by_layer = np.load(filepath).item()

        for layer, h in h_by_layer.items():
            logging.info("Computing SVD for %s layer %s activations of shape %s", filepath, layer, h.shape)
            # TODO: Should we be taking the multilinear SVD, or reshaping h to (1000, .)?
            # KR: I think we should flatten first b/c in the last couple layers (FC), we will already have "flat" matrices.
            # KR: We might be able to speed up by not calculating u, v if we aren't planning to use them
            u, s, vh = np.linalg.svd(h.reshape(MAX_PER_CLASS, -1).T, full_matrices=False)
            # Take the multilinear SVD, and flatten singular values
            # u, s, vh = np.linalg.svd(h, full_matrices=False)
            # s = s.flatten()

            path_full = os.path.join('singular_values_vecs', args.model)
            if not os.path.exists(path_full):
                os.makedirs(path_full)
            savefile = '{}/%s_{}_{}.npy'.format(path_full, filepath.strip('data/{}/'.format(args.model)).strip('.npy'), layer.split('/')[0])
            np.save(savefile % 'singularValues', s)
            np.save(savefile % 'singularVectors', u)
            # TODO: group s and then store


def pairwise_svd(args):
    files = glob("data/{}/cifar10_{}*.npy".format(args.model, args.split))
    print("Found %d activation files" % len(files))

    def compute(file1, file2):
        h1_by_layer = np.load(file1).item()
        h2_by_layer = np.load(file2).item()
        for layer in h1_by_layer.keys():
            path_full = os.path.join(OUT_PATH, 'pairwise_sv', args.model)
            if not os.path.exists(path_full):
                os.makedirs(path_full)
            savefile = '{}/%s_{}_{}_{}.npy'.format(path_full, file1.strip('data/{}/'.format(args.model)).strip('.npy'), file2.split('_')[-1].strip('.npy'), layer.split('/')[0])

            if os.path.isfile(savefile):
                continue

            h_total = np.vstack((h1_by_layer[layer], h2_by_layer[layer]))
            print(layer, h_total.shape)
            dim = h1_by_layer[layer].shape[0] + h2_by_layer[layer].shape[0]
            u, s, _ = np.linalg.svd(h_total.reshape(dim, -1).T, full_matrices=False)

            np.save(savefile % "singularValues", s)
            np.save(savefile % "singularVectors", u)
    files = sorted(files)
    activation_file_pairs = []
    for i, file1 in enumerate(files):
        for file2 in files[i+1:]:
            activation_file_pairs.append((file1, file2))

    if args.workers < 1:
        args.workers = multiprocessing.cpu_count()
    Parallel(n_jobs=args.workers)(delayed(compute)(file1, file2) for file1, file2 in activation_file_pairs)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Launcher for datadim experiments")
    parser.add_argument("task", metavar="T", type=str)
    parser.add_argument("--seed", default=1234, required=False, type=int)
    parser.add_argument("--bs", default=10, required=False, type=int)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model", choices=["vgg", "mlp5", "mlp8", "mlp12"], default="vgg")
    parser.add_argument("--workers", default=-1, type=int)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.task in globals():
        task_function = globals()[args.task]
        task_function(args)

