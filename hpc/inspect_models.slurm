#!/usr/bin/bash
#SBATCH --job-name="shappath"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=20
#SBARCH --mem=40gb
#SBATCH --time=200:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./inspect_models.py 19 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/shap_pathdep_train_q08
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
