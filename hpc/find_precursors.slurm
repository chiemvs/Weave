#!/usr/bin/bash
#SBATCH --job-name="partial"
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=125gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 62 $HOME/processed $HOME/clusters $HOME/cluster_spearmanpar
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
