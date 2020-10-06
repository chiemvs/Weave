#!/usr/bin/bash
#SBATCH --job-name="bootsclean"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=22
#SBARCH --mem=60gb
#SBATCH --time=48:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../hpc/bootstrap_classifier_skills.py 21 $HOME/clusterpar3_roll_spearman_varalpha $HOME/bootstrap_clean
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
