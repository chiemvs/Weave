#!/usr/bin/bash
#SBATCH --job-name="dimred"
#SBATCH --nodes=1
#SBATCH --nodelist=node240
#SBATCH --cpus-per-task=64
#SBARCH --mem=123gb
#SBATCH --time=04:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./dimreduce_precursors.py 62 $HOME/processed $HOME/clusters $HOME/spatcov $HOME/spatcov
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
