#!/usr/bin/bash
#SBATCH --job-name="dbsst"
#SBATCH --nodelist=node241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=123gb
#SBATCH --time=02:40:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 29 $HOME/paramtest/dbscan/eps1300_ms2000
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
