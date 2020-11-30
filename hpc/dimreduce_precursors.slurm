#!/usr/bin/bash
#SBATCH --job-name="z300dimred"
#SBATCH --nodes=1
#SBATCH --nodelist=node240
#SBATCH --cpus-per-task=64
#SBARCH --mem=125gb
#SBATCH --time=120:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./dimreduce_precursors.py 64 $HOME/processed $HOME/clusters $HOME/clusters_cv_spearmanpar_varalpha $HOME/clusters_cv_spearmanpar_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
