# ITP: Instance-Aware Test Pruning for Out-of-Distribution Detection

## Dataset Preparation

### 1. CIFAR Benchmarks 

Unzip and place the following datasets into the directory  `./cifar_datasets`.

#### In-distribution dataset

- CIFAR-10:   https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

- CIFAR-100:   https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

#### Out-of-distribution dataset

- SVHN:   http://ufldl.stanford.edu/housenumbers/test_32x32.mat

- Textures:   https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

- Places365:           http://data.csail.mit.edu/places/places365/test_256.tar

- LSUN-C:        https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz

- LSUN-R:      https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz

- iSUN:      https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz

### 2. ImageNet Benckmark 
Unzip and place the following datasets into the directory  `./imagenet_datasets`.

#### In-distribution dataset

- Imagenet-1k:   https://image-net.org/challenges/LSVRC/2012/index

#### Out-of-distribution dataset

- iNaturalist:   http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz

- SUN:           http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz

- Places:        http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz

- Textures:      https://www.robots.ox.ac.uk/~vgg/data/dtd/



## Pre-trained Model Preparation

For CIFAR, the model we used in the paper is already in the checkpoints folder. 

For ImageNet, the model we used in the paper is the pre-trained ResNet-50 provided by TorchVision library https://download.pytorch.org/models/resnet50-0676ba61.pth.

## Preliminaries
It is tested under Ubuntu Linux 18.04 and Python 3.8.19 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [numpy](http://www.numpy.org/)

## Run

### 1. Parameter Contribution Distribution Estimation

```
cd scripts
sh get_distribution.sh <DATASET> <MODEL> <GPU_ID>
```

### 2. Instance-Aware Test Pruning

```
sh test.sh <DATASET> <MODEL> <METHOD> <GPU_ID>
```

