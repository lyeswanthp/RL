#!/bin/bash
#SBATCH --job-name=mimic_preprocess
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=b.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # Request 16 CPU cores for parallel processing
#SBATCH --mem=128G                  # Request 128GB RAM (MIMIC-IV needs lots of memory)
#SBATCH --time=24:00:00             # Allow up to 24 hours
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lovelyyeswanth2002@gmail.com

# Load any required modules (adjust based on your cluster)
# module load python/3.9
# module load anaconda3

# Activate your conda/virtual environment if needed
# source activate your_env_name

# Change to project directory
cd /home/user/RL

# Run preprocessing with all CPU cores
echo "Starting MIMIC-IV preprocessing at $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 128GB"

python run_preprocessing.py

echo "Preprocessing completed at $(date)"
