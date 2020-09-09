#!/usr/bin/bash
#SBATCH --job-name="snowsi"
#SBATCH --nodes=1
#SBATCH --nodelist=node241
#SBATCH --cpus-per-task=21
#SBARCH --mem=120gb
#SBATCH --time=120:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./dimreduce_precursors.py 20 $HOME/processed $HOME/clusters $HOME/clusterpar3_roll_spearman_varalpha $HOME/clusterpar3_roll_spearman_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
