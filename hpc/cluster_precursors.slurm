#!/usr/bin/bash
#SBATCH --job-name="ms600mcs2000"
#SBATCH --nodelist=node241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30gb
#SBATCH --time=01:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_precursors.py 19 $HOME/paramtest/hdbscan/ms600_mcs2000
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
