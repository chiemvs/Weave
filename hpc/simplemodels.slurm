#!/usr/bin/bash
#SBATCH --job-name="eofs"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --mem=50gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../scripts/simplemodels.py 16 $HOME/
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
