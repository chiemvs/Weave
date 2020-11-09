#!/usr/bin/bash
#SBATCH --job-name="partial"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --mem=115gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 39 $HOME/processed $HOME/clusters $HOME/cluster_spearmanpar_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
