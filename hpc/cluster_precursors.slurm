#!/usr/bin/bash
#SBATCH --job-name="testclust"
#SBATCH --nodelist=node240
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60gb
#SBATCH --time=20:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 31 $HOME/clustertest_roll_spearman_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
