#!/usr/bin/bash
#SBATCH --job-name="alpha00005"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --mem=123gb
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./find_precursors.py 31 $HOME/processed $HOME/clusters $HOME/paramtest/alpha000005
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
