#!/usr/bin/bash
#SBATCH --job-name="cluster"
#SBATCH --nodelist=node240
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40gb
#SBATCH --time=80:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 9 $HOME/clusters_cv_spearman_varalpha_strict
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
