#!/usr/bin/bash
#SBATCH --job-name="dimred"
#SBATCH --nodes=1
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=30
#SBARCH --mem=120gb
#SBATCH --time=01:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./dimreduce_precursors.py 25 $HOME/processed $HOME/clusters $HOME/paramtest/dbscan/eps1200_ms1000 $HOME/paramtest/dbscan/eps1200_ms1000
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
