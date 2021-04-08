#!/usr/bin/bash
#SBATCH --job-name="bs_hyb"
#SBATCH --nodes=1
#SBATCH --nodelist=node240
#SBATCH --cpus-per-task=11
#SBARCH --mem=40gb
#SBATCH --time=220:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./bootstrap_classifier_skills.py 11 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/bootstrap_hybrid_noncv
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
