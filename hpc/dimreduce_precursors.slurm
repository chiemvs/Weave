#!/usr/bin/bash
#SBATCH --job-name="dimred"
#SBATCH --nodes=1
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=24
#SBARCH --mem=110gb
#SBATCH --time=120:00:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./start.sh ../requirements.txt ./dimreduce_precursors.py 23 $HOME/processed $HOME/clusters $HOME/clusters_cv_spearmanpar_varalpha_strict $HOME/clusters_cv_spearmanpar_varalpha_strict
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
