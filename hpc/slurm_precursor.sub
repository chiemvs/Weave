#!/usr/bin/bash
#SBATCH --job-name="spearman"
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=120gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 31 $HOME/processed $HOME/clusters $HOME/correlation_roll_spearman
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
