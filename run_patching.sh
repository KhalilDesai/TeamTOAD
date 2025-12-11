#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=32       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 03:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=40GB        # INCREASED from 16GB to 32GB
#SBATCH --job-name='patching'

#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err

#SBATCH --mail-type=END         # Sends email when the job finishes
#SBATCH --mail-user=your_email@example.com   # Replace with your email

# loads and sources conda module
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh

# activates environment
conda activate final_project

# runs patching command
python our_preprocessing/create_patches_fp.py --source TOAD_dataset_validation --save_dir patches_dir_validation --patch_size 256 --seg --patch --stitch 