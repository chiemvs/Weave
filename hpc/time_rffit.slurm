#!/usr/bin/bash
#SBATCH --job-name="t100000"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=32
#SBATCH --time=15:00:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./start.sh ../requirements.txt ./time_rffit.py 32 100000
# ./start.sh requirements.txt test_script.py NPARPROCS TOTAL_PREDICTORS
