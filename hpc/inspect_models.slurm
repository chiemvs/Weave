#!/usr/bin/bash
#SBATCH --job-name="permtrain"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBARCH --mem=80gb
#SBATCH --time=300:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./inspect_models.py 20 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/permimp_train_q08
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
