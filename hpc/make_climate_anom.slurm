#!/usr/bin/bash
#SBATCH --job-name="clim"
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=24
#SBATCH --time=30:50:00
#SBATCH --partition=ivm
#SBATCH --output=/scistor/ivm/jsn295/python_%j.out
#SBATCH --error=/scistor/ivm/jsn295/python_%j.err

./start.sh ../requirements.txt ./make_climate_anom.py 23 $HOME/ERA5 $HOME/processed
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
