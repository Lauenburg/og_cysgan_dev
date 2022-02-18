# CCGAN Pipeline

## Quick start

1. Clone the [Pytorch Connectomics](https://github.com/zudi-lin/pytorch_connectomics) in to the root folder of this repo.
2. Create a dataset folder with the following structure in `cerberus/ccgan`.
    ```
    ├── datasets
        └── dorsal_crop_3D_full
            ├── gt_seg_mask
            ├── testA
            ├── testB
            ├── trainA
            └── trainB
    ```
3. Populate the dataset folder with the provided data:
    ```
    .
    └── dorsal_crop_3D_full
        ├── gt_seg_mask
        │   └── full_volume_seg_3D_droped_85_75_80.tif
        ├── testA
        │   └── full_volume_em_3D_droped_85_75_80.tif
        ├── testB
        │   └── full_volume_em_3D_droped0_0_0_tile_size_17_65_65.tif
        ├── trainA
        │   └── full_volume_em_3D_droped_85_75_80.tif
        └── trainB
            └── full_volume_em_3D_droped0_0_0_tile_size_17_65_65.tif
    ```
4. Go to `cerberus/slurm_jobs`.
5. Start a `visdom` server by running: `sbatch --time=0-30:00 --export=port=9999 visdomserver.sbatch`
5. Start the pipline by running:
```
sbatch --time=1200 --job-name='ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65' \
--output="./slurm_out/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.out" \
--error="./slurm_err/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.err" \
cerberus_pipline.sbatch \
"/n/pfister_lab2/Lab/leander/cerberus/slurm_jobs/.default_configs/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.yaml"
```


The pipeline sets up a folder with the experiment name defined in the pipeline config. 
After the pipeline fully executed the experiment folder will have the following structure:

```bash
├── <experiment_name>
    ├── results
    │   └── <ccgan_model_name>
    │       └── test_latest
```

**Tip**: After running the pipeline once set config `install: 0` to `install: 1` in the config file `ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.yaml`. This will simply activate the conda env instead of setting it up again

## Configure Experiment

Except for the SLURM job specific configurations (scroll down to the SLURM chapter) the configurations are handled via a single pipeline config file. If no config file is specified the default file `em2exm/slurm_jobs/.default_configs/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.yaml` is used. You can specify your own config file as the first argument `$1`. The arguments from the config yambl are processed via the helper function `parse_yaml` implemented in `utils.sh`. The script expects you to start the pipeline from with in the `slurm_job` folder and the `utils.sh` file to be placed in its root. If this is not the case you can defined the path to the utils script via the second argument `$2`.

Example:
```bash
sbatch cerberus_pipline.sbatch path/to/config.yaml /path/to/utils.sh
```

If we want to monitor the standard output and error output be aware that for the default configurations all jobs will write to the same two files `./slurm_out/pipline_output.out` and `./slurm_err/pipline_errors.err`. For multiple jobs it is advisable to either add `%j` in to the file names (e.g. `./slurm_err/pipline_errors_%J.err`), to include the job number, or to specify the file names via the sbatch optinos `-error` and `-output`. 

Example:
```bash
sbatch --time=1400 --job-name='ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65' --output="./slurm_out/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.out" --error="./slurm_err/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.err" 
ccgan_pipline.sbatch "/n/pfister_lab2/Lab/leander/em2exm/slurm_jobs/.default_configs/ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65.yaml"
```


### Configuration overview

All configurations are managed via the python module `argparse` and are either defined in `base_option.py`, `test_option.py`, `train_option.py`, which can be found in the `options` folder of the `ccgan` folder, or at the top of `/n/pfister_lab2/Lab/leander/em2exm/ccgan/data/cerberus_dataset.py` and `/n/pfister_lab2/Lab/leander/em2exm/ccgan/models/cycle_gan_model.py`. 
However the most important settings can be configured in the experiment config file.

List and explanation of the different configurations that should configured

* experiment:
    * name : name of the experiment
    * install 0 : 0 | 1 - For 0 installs requirements, set up conda env, start conda env. For 1 starts conda env.
    * ccgan_train: 0 | 1 - Should be set to 0 when training. Should be set to 1 for in inferencing
    * ccgan_infer: 0 | 1 - Should be set to 1 when training. Should be set to 0 for in inferencing
    * blending: 0 | 1 -  Merge the created output sub-volumes using Zudi's gaussian blending
* ccgan:
    * model_name : Name of the model - used when e.g. creating the folder structure. Give it the same name as the config file
    * epochs : Number of epochs
    * epochs_decay : Number of additional epochs during which the learning rate decays linear to zero
    * display_server : Server address for the visdom server e.g. 'http://holygpu7c1701.rc.fas.harvard.edu' 
    * display_port : Display port of the visdom server e.g. '9999'
    * model_mode : 
        * --cerberus : If set - Running the cerberus configuration
        * --netG cerberus : Using the 3DUnet with 3 or 4 channel out put depending on --bcd being set or not
        * detach_advers : If set - Creates a version of B_{B,r}, B_{C,r}, B_{D,r} that is detached from G_B and uses them for the pseudo cycle consistency
        * detach_pcycle : If set - Creates a version of B_{B,r}, B_{C,r}, B_{D,r} that is detached from G_B and uses them for the additional adverserial loss
    * data_mode : 
        * --dataset_mode cerberus : Using the dataloader cerberus 
        * --data_dim 3 : expecting the data to be 3D 
        * --bcd : If set - The dataloader also loads and the model uses the distance map 
        * --vs_z 17 : sub-volume dimension z
        * --vs_x 65 : sub-volume dimension x
        * --vs_y 65 : sub-volume dimension y
    * model_mode_train
        * --save_epoch_freq 5 : frequency at which to save the model, default is 5
    * data_mode_train: 
        * --batch_size 4 : Number of volumes per patch'
        * --ss_z 1 : stride length in direction z
        * --ss_x 1 : stride length in direction x
        * --ss_y 1 : stride length in direction y
    * weigths_train: 
        * --lambda_identity 0.5 : Weight for the identity loss between A=G_B(A) and B=G_A(B), 0.5 is the default 
        * --cerberus_contour_weight 3 : Weight for the contour map cycle consistency loss (A->B-A)
        *  --cerberus_mask_weight 2 : Weight for the mask cycle consistency loss (A->B-A)
        *  --cerberus_distance_weight 2 : Weight for the distance map cycle consistency loss (A->B-A)
        *  --cerberus_D_weight_syn 1 : Weight for the mask based adversarial loss, synthesized
        *  --cerberus_D_weight_rec 1 : Weight for the mask based adversarial loss, reconstructed
        *  --lambda_B_BCD 2 :  weight for (cycle) loss  A_C ≈ G_A(B_C) with (B,B_M,B_C)-> (A,A_M,A_C) -> (B,B_M,B_C )
    * data_files_train: 
        * --max_dataset_size 500 : Max size of the dataset
        * --tif_A_file_name full_volume_em_3D_droped_85_75_80.tif : EM train file
        * --tif_A_file_name_label full_volume_seg_3D_droped_85_75_80.tif : EM train label file
        * --tif_B_file_name full_volume_em_3D_droped0_0_0_tile_size_33_65_65.tif : ExM train file
    * data_files_train: 
        * --max_dataset_size 9999 : Max size of the dataset - requires high value to process all images
        * --tif_A_file_name full_volume_em_3D_droped_85_75_80.tif : EM train file
        * --tif_A_file_name_label full_volume_seg_3D_droped_85_75_80.tif : EM train label file
        * --tif_B_file_name full_volume_em_3D_droped0_0_0_tile_size_33_65_65.tif : ExM train file
    * data_mode_infer: 
        * --ss_z 8 : stride length in direction z
        * --ss_x 32 : stride length in direction x
        * --ss_y 32 : stride length in direction y
    * data_vol_save:
        * --save_vol_A 0000000 : A string of seven binary values indicating which volumes to save, Order of volumes: fake_B, seg_syn_mask_A, seg_syn_contours_A, seg_rec_mask_A, seg_rec_contours_A, seg_syn_distance_A, seg_rec_distance_A
        * --save_vol_B 0000000 : A string of seven binary values indicating which volumes to save, Order of volumes: fake_A, seg_syn_mask_B, seg_syn_contours_B, seg_rec_mask_B, seg_rec_contours_B, seg_syn_distance_B, seg_rec_distance_B
    * num_test: Number of tests that should be run - requires high value to process all images

#### Additional configurations


## Multiple experiments

To start multiple experiments copy the configurations file, adjust the file name, change the name of the experiment, the name of the file, and which ever hyperparameter is to be investigated.

```bash
experiment:
  name : 'ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65' 

ccgan:
  model_name : 'ccgan__dorsal_crop_3D_full__Cerb2__25E_15_DE__vs_17_65_65'
```

Additionally you may want to start a separate visdom server - else the plots become meaningless.
For how to start a new visdom server scroll down to the Visdom subsection. Then update the config file. 
```bash
ccgan:
  display_server : 'http://holygpu7c1706.rc.fas.harvard.edu' 
  display_port : '9998'
```

Remember to also update the slurm job command:

```bash
sbatch --time=1400 --job-name='<updated_job_name>' 
--output="./slurm_out/<updated_out_file_name>.out"
--error="./slurm_err/<updated_err_file_name>.err" 
ccgan_pipline.sbatch "/n/pfister_lab2/Lab/leander/em2exm/slurm_jobs/.default_configs/<updated_config_file_name>.yaml"
```

## Slurm
CylceGAN is executeted using sbatch file `sbatch cerberus_pipline.sbatch`.

As SLURM interprets arguments as string literals we have to pass the SLURM specific arguments using the SLURM option which can be viewed via `sbatch --help`. Alternatively, we can overwrite them directly in `cerberus_pipline.sbatch`. The SLURM specific configurations are defined at the top of the `sbatch cerberus_pipline.sbatch` script and marked with an `##SBATCH` at the beginning. 

Important settings:
The training process requires 4 workers:
```bash
#SBATCH -c 4                
```
Each worker in turn needs a GPU
```bash
#SBATCH -p gpu_requeue      # Partition to submit to
#SBATCH --gres=gpu:4	    # Request a gpu
```

## Visdom

CycleGAN depends on a visdom server to run for the visualization. To start a visdom server you can run:

```bash 
$ module load python/3.8.5-fasrc01
$ pip install visdom 
$ python -m visdom.server -port 9999
```

To start the visdom server in the background it is advisable to use `screen` (see `screen --help`) or similar.
```bash
$ screen -S visdom
$ python -m visdom.server -port 9999
```

Starting the visdom server will return an address on which the server runs (*Something like* `http://holylogin02.rc.fas.harvard.edu`). In the pipeline config update the `display_server` and `display_port` arguments with the information from the visdom server.

To view the visdom board use an ssh tunnel for port forwarding, like:
```bash
ssh -L 8000:localhost:9999 lauenburg@holylogin02.rc.fas.harvard.edu
```
- `holylogin02` has to be updated with the correct login server on which you started the visdom server
- `8000` refers to the port on your host machine
- `9999` is the port used by the visdom board on the login server

**Important**: 
If you ran multiple experiments the visdom application can become rather heavy and considered a computational expensive application by the admin of your cluster. It is therefore advisable to also start this service using a slurm job. To run the visdom server using a slurm job run `sbatch --time=0-12:00 --export=port=9999 visdomserver.sbatch`. You can find the server address by running `squeue`.

## Utils 

The utils.sh script contains a single function that is used to be able to handle `.yaml` files as configs.

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