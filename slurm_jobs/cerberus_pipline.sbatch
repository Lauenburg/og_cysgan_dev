#!/bin/bash
#SBATCH -c 4                 # Number of cores (-c)
#SBATCH -t 0-14:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p cox               # Partition to submit to
#SBATCH --gres=gpu:7	     # Request a gpu -> do NOT set to zero or scripts will fail
#SBATCH --mem=200000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./slurm_out/pipline_output.out  # File to which STDOUT will be written - add %j to the name to include the jobid
#SBATCH -e ./slurm_err/pipline_errors.err  # File to which STDERR will be written - add %j to the name to include the jobid

# Send an email to my email account when the job is done
#SBATCH --mail-type=FAILED,END
#SBATCH --mail-user=leander.lauenburg@gmail.com 

echo "#############SETUP#############"
echo ""

# Load bash util script to handle yaml config
# Load the configs
echo "=============UTILS=============" 
echo "The script takes two arguments:"
echo "  1: Path to the utils.sh script    -> defaults to ./utils.sh"
echo "  2: Path to the pipline configfile -> defaults to ./.default_configs/test_pipline.yaml"

util="./utils.sh"
config="./.default_configs/test_pipline.yaml"

# Make the parse_yaml function in utils.sh excutable
source "${2:-$util}"
# Retrive the configurations set in the pipline yaml as variables in to this script
# The paths to the config script has to be pased as the first argument.
eval $(parse_yaml ${1:-$config})

# setup the experiment specific folder structure add the copy of the config file to the 
echo "=============EXPERIMENT=============" 
echo "Setting up the experiment specific folder structure"
mkdir -p "${experiment_base_dir}${experiment_slurm}$experiment_name/ccgan/results"
mkdir -p "${experiment_base_dir}${experiment_slurm}$experiment_name/ccgan/checkpoints"
#mkdir -p "${experiment_base_dir}${experiment_slurm}$experiment_name/ccgan"
echo "Saving a copy of the pipline config file to the experiment folder"
cp ${1:-"${experiment_base_dir}${experiment_slurm}/.default_configs/test_pipline.yaml"} "${experiment_base_dir}${experiment_slurm}${experiment_name}/test_pipline_${ccgan_model_name}.yaml"

echo ""
echo "#############CCGAN#############"
echo ""

echo "==========DEPENDENCIES========="

echo "Install cuda and python modules"

module load cuda/11.1.0-fasrc0 cudnn/8.0.4.30_cuda11.1-fasrc01
module load python/3.8.5-fasrc01
module load GCC/8.2.0-2.31.1

echo "Change working directory to: " 
cd "${experiment_base_dir}${experiment_pytc}"
echo "$(pwd)"

echo "Seting up conda env for pytorch_connectomics"
echo "Only install again if install == 0 or not defined"
module load Anaconda/5.0.1-fasrc02
# install, setup and activate env
if [ ${experiment_install:-"0"} == "0" ]
then
conda create -n py3_torch python=3.8
source activate py3_torch

conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia


echo "Installing pytorch_connectomics"
pip install --upgrade pip
pip install --editable .

echo "Change working directory to: "
cd "${experiment_base_dir}${experiment_ccgan}"
echo "$(pwd)"

echo "Installing ccgan python dependencies"
pip install -e . 
pip install -r requirements.txt
else
# only activate env
source activate py3_torch
echo "Change working directory to: "
cd "${experiment_base_dir}${experiment_ccgan}"
fi

echo "=============TRAIN============="
if [ ${experiment_ccgan_train:-"0"} == "0" ]
then
echo "Train a CCGAN model using the configuratons defined in the pipline config file under ccgan"
python train.py \
--name $ccgan_model_name --model cycle_gan \
--dataroot "${experiment_base_dir}${experiment_ccgan}${ccgan_data_root}" \
--checkpoints_dir "${experiment_base_dir}${experiment_slurm}${experiment_name}/ccgan/checkpoints" \
--n_epochs $ccgan_epochs --n_epochs_decay $ccgan_epochs_decay \
--display_server $ccgan_display_server --display_port $ccgan_display_port \
--display_ncols $ccgan_display_columns \
$ccgan_model_mode \
$ccgan_data_mode \
$ccgan_display_train \
$ccgan_model_mode_train \
$ccgan_data_mode_train \
$ccgan_weigths_train \
$ccgan_data_files_train 
echo ""
fi

echo "=============INFER============="
if [ ${experiment_ccgan_infer:-"0"} == "0" ]
then
echo "Inferencing the testA and testB dir"
python test.py --name $ccgan_model_name  --model cycle_gan \
--dataroot "${experiment_base_dir}${experiment_ccgan}${ccgan_data_root}" \
--checkpoints_dir "${experiment_base_dir}${experiment_slurm}${experiment_name}/ccgan/checkpoints" \
--results_dir "${experiment_base_dir}${experiment_slurm}${experiment_name}/ccgan/results/" \
--phase test --num_test $ccgan_num_test \
$ccgan_model_mode \
$ccgan_data_mode \
$ccgan_data_stride_infer \
$ccgan_data_files_infer \
$ccgan_data_vol_save 
echo ""
fi

echo "=============GAUS_BLENDING============="
if [ ${experiment_blending:-"0"} == "0" ]
then
python ../img_toolbox/gaussian_blending.py \
--output_dir "${experiment_base_dir}${experiment_slurm}${experiment_name}/ccgan/results/${ccgan_model_name}/test_latest/images/" \
$blending_data_vol_save \
$blending_merged_vol_A_shape \
$blending_merged_vol_B_shape \
$blending_sub_vol_shape \
$blending_stride
fi

echo "All done"