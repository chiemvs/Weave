#!/usr/bin/bash

if [ ! $# = 5 ]
then 
 echo "call this script with python package requirements file and script to execute and input and output directory"
 echo "./start.sh requirements.txt NPARPROCS test_script.py INDIR OUTDIR"
 exit
fi

NPROC=$2
SCRIPT=$3
INDIR=$4
OUTDIR=$5

module purge
module load python/3.7.4
python3 -m venv $TMPDIR/VENV
source $TMPDIR/VENV/bin/activate
pip install -r $1

python3 $SCRIPT $TMPDIR $INDIR $HOME $NPROC $OUTDIR # script, tmpstorage_on_the_node, obsdir (reading data), packagedir (for loading the code), nprocesses, OUTDIR (writing results)

#rsync -av $TMPDIR/ $HOME --exclude VENV --exclude nodelist
cp $TMPDIR/*.log $HOME
cp $TMPDIR/*.dat $HOME
