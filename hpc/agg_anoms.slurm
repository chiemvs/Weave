#!/usr/bin/bash
#SBATCH --job-name="sstagg"
#SBATCH --nodelist=node015
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --partition=defq
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./start.sh ../requirements.txt ./agg_anoms.py 11 $HOME/processed $HOME/processed_resample 
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
