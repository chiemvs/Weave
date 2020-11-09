#!/usr/bin/bash
#SBATCH --job-name="clustsst"
#SBATCH --nodelist=node241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60gb
#SBATCH --time=40:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 7 $HOME/clustertest
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
