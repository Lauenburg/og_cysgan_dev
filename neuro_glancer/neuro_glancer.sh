#!/bin/bash
#SBATCH -c 8                # Number of cores (-c)
#SBATCH -t 0-05:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p cox #gpu_requeue      # Partition to submit to
#SBATCH --gres=gpu:4	    # Request a gpu
#SBATCH --mem=100000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./slurm_out/neuro_glancer_output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./slurm_err/neuro_glancer_errors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Send an email to my email account when the job is done
#SBATCH --mail-type=FAILED,END
#SBATCH --mail-user=leander.lauenburg@gmail.com 

# Retrive information on the memory and time that would be needed to run this job
##SBATCH --test-only

echo "=========================="
echo "Install necessary modlues"
module load Anaconda3/2020.11
source ~/.bashrc

#conda create -n py3_torch python=3.8
source activate py3_torch
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch

pip install --upgrade pip
pip install neuroglancer imageio h5py cloud-volume

#conda install libgcc -y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home05/lauenburg/.conda/envs/img_toolbox/lib/
export XDG_RUNTIME_DIR=""

echo "=========================="
echo "Installing python dependencies"
pip install -r requirements.txt
echo "=========================="
echo "Starting jupyter nootbook"
python -i neuro_glancer.py