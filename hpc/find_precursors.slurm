#!/usr/bin/bash
#SBATCH --job-name="e-6-12"
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=64
#SBATCH --time=40:00:00
#SBATCH --mem=125gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 62 $HOME/processed $HOME/clusters $HOME/paramtest/varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
