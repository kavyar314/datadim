# Dimen

Exploring how neural networks compress data dimensionality.

## Setup

First, download `cifar10vgg.h5` weights at https://drive.google.com/open?id=0B4odNGNGJ56qVW9JdkthbzBsX28, `mkdir data`, and place the `cifar10vgg.h5` file in the `data` folder.

## Running inference

```
python datadim.py infer --split train --bs 40
python datadim.py infer --split test --bs 40
```
