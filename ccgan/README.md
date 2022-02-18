# CustomCycleGAN aka CCGAN

This repo is based on the original [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repo. The data loader and models were adapted for domain transfer from electron microscopy to expansion microscopy for 3D volumes representing the nervous system. 

The to train and test the network we use the `train.py` and `test.py` scripts.

## Overview

```bash
├── ccgan
    ├── data
    ├── datasets
    ├── models
    ├── options
    ├── scripts
    ├── tests
    └── util
```

### Data

Contains the dataloader. All dataloader are derived from the `base_dataset.py`. `unaligned_dataset.py` is the default dataloader when training the CycleGAN. `single_dataset.py` can be when testing to only transfer images from A to B (which domain is A and which is B can be set using the options). 

The `__init__.py` file serves to initialize a dataloader based on the options settings. 

#### `unaligned_dataset.py`

**Segmentation masks matching**

#TODO: **Document the changes of:**
- [ ] `get_transform` in `base_dataset.py`
- [ ] `UnalignedDataset` in `unaligned_dataset.py`

### Datasets

Holds the datasets for the domain transfer. 

### Models

Contains the CycleGAN and its building blocks. The CycleGAN implementation is based on the `base_model.py` parent class. `cycle_gan_model.py` initializes the networks, loads the data, defines the forward, backward, and optimization step. `networks.py` creates and defines the networks. `test_models.py` is used to derive test results for only one direction. It is used together with `single_dataset.py` from the `data` folder.

#### `cycle_gan_model.py`

**Segmentation masks matching**

#TODO: **Document the changes of:**
- [ ] Forward pass
- [ ] Optimization
- [ ] Initialization

#### `networks.py`

**Segmentation masks matching**

#TODO:
- [ ] Document the changes of ResnetGenerator

### Options

The folder contains a `train_options.py` and a `test_options.py`n script that both inherit from the `base_option.py` script and are used to efficiently configure experiments. 

#TODO: 
- [ ] Describe adaptation of `train_options.py`
- [ ] Describe adaptation of `test_options.py`

### Scripts

#TODO: ### **Describe scripts**

### Tests

#TODO: ### **Describe tests**


### Util

The util folder contains the following utility scripts:

#### `image_pool.py`
This class implements an image buffer that stores previously generated images.

#### `single_html.py`
This HTML class allows us to save images and write texts into a single HTML file.

#### `utils.py`
This script contains multiple util functions used for transferring data, saving images, testing connections, and handling folder structures

#### `visualizer.py`
Handles the visdom based visualization of the results and training process.