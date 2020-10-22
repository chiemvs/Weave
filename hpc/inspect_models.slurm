#!/usr/bin/bash
#SBATCH --job-name="shaptrain"
#SBATCH --nodes=1
#SBATCH --nodelist=node015
#SBATCH --cpus-per-task=32
#SBARCH --mem=40gb
#SBATCH --time=200:00:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ../scripts/inspect_models.py 31 $HOME/clusterpar3_roll_spearman_varalpha $HOME/shaptest_negative_train
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
