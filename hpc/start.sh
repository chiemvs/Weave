#!/usr/bin/bash

if [ ! $# = 2 ]
then 
 echo "call this script with python package requirements file"
 echo "./start.sh requirements.txt NPARPROCS"
 exit
fi

NPROC=$2

module purge
module load python/3.7.4
python3 -m venv $TMPDIR/VENV
source $TMPDIR/VENV/bin/activate
pip install -r $1

python3 parclust.py $TMPDIR $HOME $HOME $NPROC # tmpstorage_on_the_node, obsdir (reading data, writing results), packagedir (for loading the code), nprocesses

#cp $TMPDIR/*.dat $HOME/clust/
