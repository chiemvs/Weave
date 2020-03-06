#!/usr/bin/bash

if [ ! $# = 3 ]
then 
 echo "call this script with python package requirements file and script to execute"
 echo "./start.sh requirements.txt NPARPROCS test_script.py"
 exit
fi

NPROC=$2
SCRIPT=$3

module purge
module load python/3.7.4
python3 -m venv $TMPDIR/VENV
source $TMPDIR/VENV/bin/activate
pip install -r $1

python3 $SCRIPT $TMPDIR $HOME $HOME $NPROC # script, tmpstorage_on_the_node, obsdir (reading data, writing results), packagedir (for loading the code), nprocesses

#cp $TMPDIR/*.dat $HOME/clust/
