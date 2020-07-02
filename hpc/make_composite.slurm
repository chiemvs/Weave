#!/usr/bin/bash
#SBATCH --job-name="comps"
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=32
#SBATCH --time=60:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./make_composite.py 31 $HOME/processed $HOME/clusters $HOME/comp_q095
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
