#!/usr/bin/bash
#SBATCH --job-name="others"
#SBATCH --nodes=1
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=40
#SBARCH --mem=125gb
#SBATCH --time=96:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./dimreduce_precursors.py 38 $HOME/processed $HOME/clusters $HOME/clustertest_roll_spearman_varalpha $HOME/clustertest_roll_spearman_varalpha
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
