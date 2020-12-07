#!/usr/bin/bash
#SBATCH --job-name="permimp6"
#SBATCH --nodes=1
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=32
#SBARCH --mem=60gb
#SBATCH --time=200:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../scripts/inspect_models.py 31 $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/permimp_val_q0666
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
