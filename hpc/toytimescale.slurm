#!/usr/bin/bash
#SBATCH --job-name="spatcov"
#SBATCH --nodelist=node242
#SBATCH --cpus-per-task=12
#SBATCH --time=00:40:00
#SBATCH --partition=ivm
#SBATCH --output=../../python_%j.out
#SBATCH --error=../../python_%j.err

./start.sh ../requirements.txt ./toytimescale.py 11 $HOME/processed $HOME/clusters $HOME/correlation_roll_spearman $HOME/spatcov
# ./start.sh requirements.txt test_script.py NPARPROCS DIRECTORIES (can be many)
# python toytimescale.py $TEMPDIR $PACKAGEDIR $NPROC $ANOMDIR $CLUSTERDIR $PATTERNDIR $OUTDIR
