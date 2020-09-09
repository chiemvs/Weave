#!/usr/bin/bash
#SBATCH --job-name="clustfull"
#SBATCH --nodelist=node241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=60gb
#SBATCH --time=40:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 39 $HOME/clusterpar3_roll_spearman_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
