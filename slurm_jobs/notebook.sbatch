#!/bin/bash
#SBATCH -c 6               # Number of cores (-c)
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu_requeue #cox #serial_requeue #gpu_requeue      # Partition to submit to
#SBATCH --gres=gpu:4	    # Request a gpu
#SBATCH --mem=250000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./slurm_out/notebook_output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./slurm_err/notebook_train_errors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Send an email to my email account when the job is done
#SBATCH --mail-type=START,FAILED,END
#SBATCH --mail-user=leander.lauenburg@gmail.com 

# Retrive information on the memory and time that would be needed to run this job
##SBATCH --test-only
module purge
module load Anaconda3/2020.11

source ~/.bashrc

conda activate img_toolbox

cd ../pytorch_connectomics

pip install --upgrade pip
pip install --editable .

cd ../mAP_3Dtoolbox

conda install --yes --file requirements.txt
pip install -r requirements.txt

cd ../img_toolbox
pip install -r requirements

jupyter notebook --no-browser --port=8888

