#!/usr/bin/bash
#SBATCH --job-name="partial"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=62
#SBATCH --time=120:00:00
#SBATCH --mem=125gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 61 $HOME/processed $HOME/clusters $HOME/correlation_cv_spearmanpar_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
