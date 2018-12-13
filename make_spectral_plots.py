#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import namedtuple
import logging
from glob import glob
import os

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy
import seaborn as sns

sns.set_context('talk')


CIFAR10_LABEL_NAMES = ['airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

PATH_TO_SV = "./singular_values"


Configuration = namedtuple('Configuration', 'path dataset mode cls layertype layerindex')


def load_singular_values(path, normalize=True):
    try:
        singular_vals = np.load(path)
        if normalize:
            singular_vals = singular_vals / singular_vals[0]
        return singular_vals
    except:
        print("Loading error on path", path)
        return None


def parse_singular_value_path(path):
    filename = os.path.basename(path).rstrip('.npy')
    _, dataset, mode, cls, layertype, layerindex = filename.split("_")
    layerindex = int(layerindex)

    cls = int(cls.strip('c'))

    return Configuration(path, dataset, mode, cls, layertype, layerindex)


def parse_singular_value_paths(model_name, match_name=None):
    # Find all singular value .npy files
    glob_pattern = os.path.join(PATH_TO_SV, model_name, "*.npy")
    sv_files = glob(glob_pattern)

    # Filter to those of the queried pattern
    if match_name:
        sv_files = filter(lambda path: match_name in path, sv_files)

    parsed = map(parse_singular_value_path, sv_files)
    sort = sorted(parsed, key=lambda c: c.layerindex)
    return sort


def plot_spectra_by_layer(model_name, cls, normalize=True, output=None):
    configurations = parse_singular_value_paths(model_name, "_"+cls+"_")

    fig = plt.figure(figsize=(8, 6))
    for configuration in configurations:
        label = "{0.layertype} {0.layerindex}".format(configuration)

        sv = load_singular_values(configuration.path, normalize=normalize)
        color = plt.cm.viridis(configuration.layerindex / 13)
        plt.semilogy(sv, label=label, c=color)

    plt.title("Singular value spectra, {}".format(model_name.upper()), pad=10)
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value magnitude" + (", $\sigma_i / \sigma_1$" if normalize else ""))
    plt.legend(bbox_to_anchor=(1, 1))
    fig.set_tight_layout(True)

    if output:
        plt.savefig(output)


def plot_spectra_by_layer_residual(model_name, cls, normalize=True, output=None):
    configurations = parse_singular_value_paths(model_name, "_"+cls+"_")
    svs = [load_singular_values(configuration.path, normalize=normalize) for configuration in configurations[:-1]]
    residuals = [np.median(svs[0]) - np.median(sv) for sv in svs]
    layers = range(len(residuals))
    colors = [plt.cm.viridis(configuration.layerindex / 13) for configuration in configurations[:-1]]
    labels = ["$X$"] + ["$A^{(%d)}$" % configuration.layerindex
                          for configuration in configurations[1:-1]]

    fig = plt.figure(figsize=(7.8, 8))
    for layer, residual, color, label in zip(layers, residuals, colors, labels):
        plt.scatter(layer, residual, c=color, marker="o", label=label)
    plt.axhline(y=0, linestyle="--")
    plt.xticks(layers)
    plt.title("Residual in singular value medians, {}".format(model_name.upper()), pad=10)
    plt.xlabel("Layer (post-activation)")
    plt.ylabel("Difference in median {}singular value".format("normalized " if normalize else ""))
    plt.legend(bbox_to_anchor=(1, 1))
    fig.set_tight_layout(True)

    if output:
        plt.savefig(output)

def plot_spectral_norm(model_name, output=None):
    fig = plt.figure(figsize=(8, 6))

    for cls in range(10):
        configurations = parse_singular_value_paths(model_name, "_c{}_".format(cls))

        x = []
        y = []
        for configuration in configurations:
            sv = load_singular_values(configuration.path, normalize=False)
            x.append(configuration.layerindex)
            y.append(np.max(sv))
        plt.plot(x, y, label="{}".format(CIFAR10_LABEL_NAMES[cls]))

    plt.xlabel("Layer (after activation)")
    plt.ylabel("Spectral norm of activations, $\sigma_1$")
    plt.title("Spectral norm of activation matrix, {}".format(model_name.upper()), pad=10)
    plt.legend()
    fig.set_tight_layout(True)

    if output:
        plt.savefig(output)


plot_spectral_norm("mlp12", output="plots/max-sv_mlp12.pdf")
cls = "c1"
plot_spectra_by_layer("mlp12", cls, normalize=True, output="plots/spectra-normalized_mlp12_{}.pdf".format(cls))
plot_spectra_by_layer_residual("mlp12", cls, normalize=True, output="plots/spectra-normalized_mlp12_resid_{}.pdf".format(cls))


plot_spectral_norm("vgg", output="plots/max-sv_vgg.pdf")
cls = "c1"
plot_spectra_by_layer("vgg", cls, normalize=True, output="plots/spectra-normalized_vgg_{}.pdf".format(cls))
plot_spectra_by_layer_residual("vgg", cls, normalize=True, output="plots/spectra-normalized_vgg_resid_{}.pdf".format(cls))

