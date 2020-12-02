#!/usr/bin/bash
#SBATCH --job-name="cluster"
#SBATCH --nodelist=node242
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40gb
#SBATCH --time=80:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 19 $HOME/clusters_cv_spearmanpar_varalpha_strict
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
