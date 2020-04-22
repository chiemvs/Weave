#!/usr/bin/bash
#SBATCH --job-name="clim"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=32
#SBATCH --time=00:50:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./make_climate_anom.py 30 $HOME/ERA5 $HOME/processed
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
