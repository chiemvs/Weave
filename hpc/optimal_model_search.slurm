#!/usr/bin/bash
#SBATCH --job-name="imp"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=64
#SBARCH --mem=60gb
#SBATCH --time=48:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../scripts/optimal_model_search.py 63 $HOME/importance 
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
