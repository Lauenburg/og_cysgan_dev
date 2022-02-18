# Custom CycleGAN (CCGAN)

The project aims transfer a segmented EM dataset to the ExM domain, train a segmentation model on the synthesized data, and apply the trained segmentation model to unsegmented ExM data. More precisely the pipeline runs through the following steps:

* CCGAN
    * Run pytests to ensure the current setup is configured correctly
    * Train CCGAN on EM to ExM transfer and vice versa
    * Transfer EM data to the ExM domain
* Create H5 data files from the transferred data and corresponding labels
* Segmentation
    * Train a segmentation model on the transferred data 
    * Inference testing data (returns foreground probability maps, instance contours and distance map)
* BC|BCD connect 
    * Generate segmentation masks from foreground probability maps, instance contours and distance map

The pipeline is run via the `ccgan_pipeline.sh` script in the `slurm_job` folder.
A new project is configured via a `config.yaml` file that is passed to the SLURM job as first argument. The default pipeline config file can be found at `slurm_jobs/.default_configs/test_pipline.yaml`.

## Overview
Main folder structure:

```bash
├── ccgan
├── img_toolbox
├── neuro_glancer
├── pytorch_connectomics
└── slurm_jobs
```

### CCGAN

Customized CycleGAN implementation -> See the [original repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for general information. 

#### Customization: 3D CCGAN

Adaptation that enables the model to process datasets of the format (#volume, depth, height, width).
The dataset should be split and stored in two tiff files. The tiff files in turn should be stored at '/path/to/data/trainA' for domain A and '/path/to/data/trainB' for domain B. Each tiff file should hold its dataset as a single 4D array of sub-volumes.

`vunalinged_dataset.py`:
A new dataset class that loads the data from the tiff files, applies the image transformations, and is iterable.

`base_option.py`:
Added the an option that lets you define if you try processing 2D or 3D data.

`networks.py`:
* `ResnetGenerator3D`:
Added 3D Resnet generator class

* `ResnetBlock3D`:
Added resnet block 3D class

* `NLayerDiscriminator3D`:
Added 3D NLayer discriminator class

* `define_G` and `define_D`:
Added logic to either build 2D or 3D networks

`visualizer.py`:
* `Visualizer`:
Adapted the class to visualize point clouds instead of images on the visdom server if processing 3D data.

* `save_images`
Adapted the function to save the 3D volumes as .npy's instead of trying to save them as .png files.

#### Customization: Segmentation Matching

The generators of the GAN's are given a second head which produces segmentation masks. The segmentation masks are then compared to the GT segmentation masks using BCE. Since we only have the GT segmentation masks for A this loss takes effect when synthesizing B or reconstructing A. The following adaptations were made

`networks.py`:
* `ResnetGenerator`:

Split the head of the model in to two. The model now has a segmentation head and a generator head. The segmentation had consist of the convolutional layer where the last layer uses 1x1 convolutions. The generator head has a single convolutional layer and a Tanh activation function.

`cycle_gan_model.py`
* `CycleGANModel.modify_commandline_options()`:

Added configuration option `--seg_match` and `--seg_match_weight`.

* `CycleGANModel.init()`:

Added the two losses `seg_syn_logits` and `seg_rec_logits`.
Added three additional frames to the visdom visualization, on for the GT segmentation mask, one for the generated segmentation mask from the forward pass G_B(A) = (B, S), and on for the segmentation mask from the backward pass (reconstruction) G_A(G_B(A)) = (B, S).
Setup up a segmentation loss function with `BCEWithLogitsLoss`.

* `CycleGANModel.set_input()`:

Load and save the current GT segmentation mask to the `seg_mask` variable.

* `CycleGANModel.forward()`:

Retrieve the additionally computed segmentation masks and save them to member variables

* `CycleGANModel.backward_G()`:

Calculate the BCE loss of the GT segmentation mask with the segmentation mask computed at the forward pass of A as well as the segmentation computed at the reconstruction of A both weighted with `--seg_match_weight`.

`unaligned_dataset.py`
* `UnalignedDataset.init()`:

Retrieve paths to the GT segmentation masks and setup the transformation function.

* `UnalignedDataset.__getitem__()`:

Load and transform the current GT segmentation mask. Convert the mask to a binary mask. Add the mask to the current data dictionary.

### Customization: Discriminative loss 

Which binary segmentation mask corresponds to the contours of the generated image

#TODO:
- [ ] Describe the customization


### (Image) Toolbox

Contains general utility scripts that are mainly focused on manipulating image data and image datasets.

### Neuro-Glancer

Visualize the data volumes in the Neuro-Glancer. The Neuro-Glancer creates 3D renderings from the data volumes and can overlay the source volume with its corresponding segmentation masks.

### Pytorch Connectomics

Then pipline described above is dependent on Zudi Lin's segmentation model from his [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics) repository. It is therefore required to clone the repo as `pytorch_connectomics` in to the root folder of EM2ExM repository. 

#### Tensorboard
When running the pipeline we can monitor the segmentation progress using tensorboard. 
To start tensorboard run `sbatch --time=0-12:00 --export=logdir="./<experiment_name>/segment",port=<port> tensorboard.sbatch` with in the `slurm_jobs` folder. You can find the server address by running `squeue`.

### Slurm Jobs 

This repo was created for and tested on a slurm ready computer cluster. Accordingly, the pipeline and most other services are started with slurm scripts. For more information refer to `./slurm_jobs/README.md`.

## Models

### Naming convention

Each model should be named using the following convention:
```bash
<model_name>__<data_name>__<configuration>__<hyper_parameters>
```
* List of model names
    * `ccgan`
* List of data names
    * `dorsal_crop_dense_filtered`
    * `em2exm_f0000_round_1`
* List of data configurations
    * `vanilla`
    * `enc_match`
    * `seg_match`
* List of hyper parameters (not complete)
    * `<nr epochs>E`
    * `<nr decay epochs>ED`

Example:
```bash
ccgan__dorsal_crop_dense_filtered__vanilla__20E_15DE
```

## Datasets

See `./ccgan/datasets/README.md` for more specific information.
Short intro and overview:

### Domain Transfer

CCGAN contains three datasets at `em2exm/ccgan/datasets`
```
├── datasets
    ├── dorsal_crop_dense_filtered
    ├── em2exm
    └── em2exm_f0000_round_1
```
1. em2exm
    * Folder structure
    ```bash
    └── em2exm
        ├── segGT_not_filtered
        └── trainA_not_filtered
    ```
    * Folder description

    `segGT_not_filtered` contains all (not filtered) `128x128` patches of the GT segmentation masks, scaling `[3,3]`
    `trainA_not_filtered` contains all (not filtered) `128x128` EM image patches, scaling `[3,3]`

    * Data source

        * EM: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/crop_em2.tif`
        * EM masks: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/mask_gt.h5`
        * EM GT seg. masks: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/seg_gt.h5`

2. em2exm_f0000_round_1
    * Folder structure
    ```bash
    └── em2exm_f0000_round_1
        ├── testA
        ├── testB
        ├── testS
        ├── trainA
        ├── trainB
        └── trainS
    ```
    * Folder description

        * `trainA`: EM `128x128` image patches filtered to match the target distribution `F000_Round1_c4_histeq_dn.h5`, scaling `[3,3]`
        * `trainB`: ExM `128x128` image patches filtered to match the target distribution `F000_Round1_c4_histeq_dn.h5`, scaling `[0.25,0.25]`
        * `trainS`: the to the EM patches corresponding `128x128` GT segmentation Patches
        * `testA`: copy of `trainA`
        * `testB`: copy of `trainB`
        * `testS`: copy of `trainS`

    * Data source

        * EM: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/crop_em2.tif`
        * EM masks: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/mask_gt.h5`
        * EM GT seg. masks: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/seg_gt.h5`
        * ExM: `/n/pfister_lab2/Lab/zudilin/data/Expansion/image/dorsal_6_histeq.h5`
        * Target distribution: `/n/pfister_lab2/Lab/zudilin/data/Expansion/F000/F000_Round1_c4_histeq_dn.h5`
        * Target distribution GT seg. masks: `/n/pfister_lab2/Lab/zudilin/data/Expansion/F000/F000_1_seg_pf.h5`
    
    * Documentation
        * `em2exm/ccgan/datasets/em2exm_f0000_round_1/README.md`

1. em2exm_f0000_round_1
    * Folder structure
    ```bash
    └── dorsal_crop_dense_filtered
        ├── testA
        ├── testB
        ├── trainA
        └── trainB
    ```
    * Folder description

        * `trainA` EM `128x128` image patches filtered for density
        * `trainB` ExM `128x128` image patches filtered for density
        * `testA` copy of `trainA`
        * `testB` copy of `trainB`

    * Data source

        * EM: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/crop_em2.tif`
        * EM masks: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/mask_gt.h5`
        * EM GT seg. masks: `/n/pfister_lab2/Lab/vcg_connectomics/EM/zebraFish/benchmark/seg_gt.h5`
        * ExM: `/n/pfister_lab2/Lab/zudilin/data/Expansion/image/dorsal_6_histeq.h5`

    * Documentation
        * `em2exm/ccgan/datasets/dorsal_crop_dense_filtered/README.md`

### Segmenter

All training datasets are created dynamically based on the output from a CycleGAN and using the `em2exm/img_toolbox/h5_util.py` script. `pytorch-connectomics` contains two inferencing datasets:
```bash
├── datasets
    └── inference
        ├── 256_not_filtered_dorsal_b.h5
        └── f000_Round1_dn_128_128.h5
```
1. 256_not_filtered_dorsal_b.h5

    400 individual `128x128` image patches. Every four patches belong to one slice and form an `256x256` image.

    ### Configs
    * Source: `/n/pfister_lab2/Lab/zudilin/data/Expansion/image/dorsal_6_histeq.h5`
    * Patch size: `[128,128]`
    * Sacling: `[0.25,0.25]`
    * Nr. image patches: 400

2. f000_Round1_dn_128_128.h5

    Segmented ExM data provided by Donglai Wei.

    ### Configs
    * Source: `/n/pfister_lab2/Lab/zudilin/data/Expansion/F000/F000_Round1_c4_histeq_dn.h5`
    * Corresponding GT seg. masks: `/n/pfister_lab2/Lab/zudilin/data/Expansion/F000/F000_1_seg_pf.h5`
    * Patch size: `[128,128]`
    * Sacling: `[1/16,1/16]`
    * Nr. image patches: -






