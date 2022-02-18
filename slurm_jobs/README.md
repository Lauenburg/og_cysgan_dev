# Slurm Jobs

The `.sbatch` files are SLURM job descriptions. A slurm script starts a job on a slurm ready computer cluster that processes the task defined by the script. A SLURM script is executed via the `sbatch <script_name>.sbatch`. Such a script is divided in to slurm specific configuration commands marked by `#SLURM` and process specific commands that are the same as in any bash script.

## CCGAN Pipeline

The `ccgan_pipeline` script serves as the main slurm script and runs the whole pipeline.
The pipeline comprises the following steps:

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

The pipeline is executed via `sbatch ccgan_pipeline.sbatch`. Its configuration is handled via a single pipeline config file - except for the SLURM job specific configurations. If no config file is specified the default file `em2exm/slurm_jobs/.default_configs/test_pipline.yaml` is used. You can specify your own config file as the first argument `$1`. The arguments from the config yambl are processed via the helper function `parse_yaml` implemented in `utils.sh`. The script expects you to start the pipeline from with in the `slurm_job` folder and the `utils.sh` file to be placed in its root. If this is not the case you can defined the path to the utils script via the second argument `$2`.

Example:
```bash
sbatch ccgan_pipeline.sbatch path/to/config.yaml /path/to/utils.sh
```

Since the SLURM interprets arguments as string literals we have to pass the SLURM specific arguments as using the SLURM option which can be viewed via `sbatch --help`. Alternatively, we can overwrite them directly in the script. The SLURM specific configurations are defined at the top of the `sbatch ccgan_pipeline.sbatch` script and marked with an `##SBATCH` at the beginning. 

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

If we configured the pipeline config file, as well as the SLURM configs, and we want to start multiple experiments we can simply copy the configurations file, change the name of the experiment, and which ever hyperparameter we want to investigate, and start a new SLURM job using the `sbatch ccgan_pipeline.sbatch path/to/config.yaml`. However if we want to monitor the standard output and error output be aware that for the default configurations all jobs will write to the same two files `./slurm_out/pipline_output.out` and `./slurm_err/pipline_errors.err`. For multiple jobs it is advisable to either add `%j` in to the file names (e.g. `./slurm_err/pipline_errors_%J.err`), to include the job number, or to specify the file names via the sbatch optinos `-e` and `-o`. To reduce cluttering it is further advisable to save the output and error file right in the experiment folder

Example:
```bash
sbatch --job-name='ccgan__dorsal_crop_dense__vanilla__20E_15DE' \
--output="ccgan__dorsal_crop_dense__vanilla__20E_15DE/match_seg_25E_25DE_VANILLA.out" \
--error="ccgan__dorsal_crop_dense__vanilla__20E_15DE/match_seg_25E_25DE_VANILLA.err" \
ccgan_pipline.sbatch "/n/pfister_lab2/Lab/leander/em2exm/slurm_jobs/.default_configs/ccgan__dorsal_crop_dense__vanilla__20E_15DE.yaml" 
```

The pipeline sets up a folder with the experiment name defined in the pipeline config. After the pipeline fully executed the experiment folder will have the following structure:

```bash
├── <experiment_name>
    ├── results
    │   └── <ccgan_model_name>
    │       └── test_latest
    └── segment
        └── test
```

The configuration files, segmentation configurations files, and the h5 training files for the segmentation model, as well as a sample of the transferred images along with their corresponding source images and GT segmentation masks, will be saved in the root of the experiment folder. The transfer model's specific inferencing output is saved in `slurm_jobs/<experiment_name>/results/<model_name>/test_latest/images`. The ouput of the segmentation model is saved to the `slurm_jobs/<experiment_name>/segment/test` folder.

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

## Segmentation

Then pipline described above is dependent on Zudi Lin's segmentation model from his [pytorch_connectomics](https://github.com/zudi-lin/pytorch_connectomics) repository. It is therefore required to clone the repo as `pytorch_connectomics` in to the root folder of EM2ExM repository. 

### Tensorboard
When running the pipeline we can monitor the segmentation progress using tensorboard. 
To start tensorboard run `sbatch --time=0-12:00 --export=logdir="./<experiment_name>/segment",port=<port> tensorboard.sbatch`
You can find the server address by running `squeue`.

## CCGAN Test

`ccgan_test.sh` is a simple slurm script that can be used to run the tests from the test folder with in the ccgan folder.

## CCGAN Train

`ccgan_train.sh` can be used to train and run inference of the CCGAN model.

## Utils 

The utils.sh script contains a single function that is used to be able to handle `.yaml` files as configs.

