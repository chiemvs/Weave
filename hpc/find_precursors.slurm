#!/usr/bin/bash
#SBATCH --job-name="regular"
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=64
#SBATCH --time=120:00:00
#SBATCH --mem=125gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 63 $HOME/processed $HOME/clusters $HOME/correlation_cv_spearman_varalpha_strict
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
