#!/usr/bin/bash
#SBATCH --job-name="permtrain"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=40
#SBARCH --mem=80gb
#SBATCH --time=300:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./inspect_models.py 40 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/permimp_train_q0666
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
