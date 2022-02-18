#!/bin/bash
#SBATCH -c 2                 # Number of cores (-c)
#SBATCH -t 0-12:00           # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p cox               # Partition to submit to
#SBATCH --gres=gpu:1	     # Request a gpu -> do NOT set to zero or scripts will fail
#SBATCH --mem=8000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./slurm_out/visdom_server_output.out  # File to which STDOUT will be written - add %j to the name to include the jobid
#SBATCH -e ./slurm_err/visdom_server_error.err  # File to which STDERR will be written - add %j to the name to include the jobid

# Send an email to my email account when the job is done
#SBATCH --mail-type=FAILED,END
#SBATCH --mail-user=leander.lauenburg@gmail.com 

echo "=========================="
echo "Install necessary modlues"
module load cuda/11.1.0-fasrc0 cudnn/8.0.4.30_cuda11.1-fasrc01
module load python/3.8.5-fasrc01
source ~/.bashrc

echo "=========================="
echo "Change working directory"
cd ../ccgan/
pip install --upgrade pip
pip install -e . 
pip install -r requirements.txt

echo "Starting on port ${port}."

python -m visdom.server -port=${port}

