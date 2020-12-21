#!/usr/bin/bash
#SBATCH --job-name="bootsbase"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=2
#SBARCH --mem=40gb
#SBATCH --time=120:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../hpc/bootstrap_classifier_skills.py 1 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/bootstrap_base
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
