#!/usr/bin/bash
#SBATCH --job-name="pclust"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --time=00:15:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 25 $HOME/correlation_roll_spearman
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
