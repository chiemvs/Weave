#!/usr/bin/bash
#SBATCH --job-name="btshyb"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=20
#SBARCH --mem=40gb
#SBATCH --time=120:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./bootstrap_classifier_skills.py 19 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/bootstrap_hybrid_noncv_base_noncv
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
