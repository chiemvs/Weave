#!/usr/bin/bash
#SBATCH --job-name="impq09"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=32
#SBARCH --mem=120gb
#SBATCH --time=200:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../scripts/optimal_model_search.py 31 $HOME/clusterpar3_roll_spearman_varalpha $HOME/importance_spatcov_q09
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
