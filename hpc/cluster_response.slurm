#!/usr/bin/bash
#SBATCH --job-name="clust075"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=30
#SBATCH --time=03:15:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./cluster_response.py 29 $HOME/ERA5 $HOME/clusters
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
