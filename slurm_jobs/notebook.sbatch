#!/bin/bash
#SBATCH -c 4               # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu #gpu_requeue #serial_requeue #gpu_requeue      # Partition to submit to
#SBATCH --gres=gpu:1	    # Request a gpu
#SBATCH --mem=50000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./slurm_out/notebook_output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./slurm_err/notebook_train_errors_%j.err  # File to which STDERR will be written, %j inserts jobid

# Send an email to my email account when the job is done
#SBATCH --mail-type=START,FAILED,END
#SBATCH --mail-user=leander.lauenburg@gmail.com 

# Retrive information on the memory and time that would be needed to run this job
##SBATCH --test-only

echo "=========================="
echo "Install necessary modlues"
#module load cuda/11.1.0-fasrc0 cudnn/8.0.4.30_cuda11.1-fasrc01
module load Anaconda3/2020.11
source ~/.bashrc
echo "=========================="
echo "Change working directory"
pwd
echo "Change working directory to: " 
cd ../pytorch_connectomics
echo "$(pwd)"

echo "Seting up conda env for pytorch_connectomics"
module load Anaconda/5.0.1-fasrc02
conda create -n py3_torch python=3.8
source activate py3_torch
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

echo "Installing pytorch_connectomics"
pip install --upgrade pip
pip install --editable .

cd ../img_toolbox
#conda create -n img_toolbox python=3.7.6 -y
#conda activate img_toolbox
#conda install -c anaconda jupyter -y

#conda install libgcc -y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home05/lauenburg/.conda/envs/img_toolbox/lib/
export XDG_RUNTIME_DIR=""

echo "=========================="
echo "Installing python dependencies"
pip install -r requirements.txt
echo "=========================="
echo "Starting jupyter nootbook"
jupyter notebook --no-browser --port=8888

