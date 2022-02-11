#!/usr/bin/bash

if [ ! $# -ge 4 ]
then 
 echo "call this script with python package requirements file, script to execute, nprocs and input and output directories"
 echo "./start.sh requirements.txt test_script.py NPARPROCS ARRAY_OF_DIRECTORIES"
 exit
fi

REQUIREMENTS=$1
SCRIPT=$2
NPROC=$3

module purge
module load python/3.7.4
python3 -m venv $TMPDIR/VENV
source $TMPDIR/VENV/bin/activate
pip3 install --upgrade pip
pip3 install wheel
pip3 install -r $REQUIREMENTS
pip3 install --no-deps PermutationImportance 

python3 $SCRIPT $TMPDIR $HOME/Documents/Weave $NPROC "${@:4}" # script, tmpstorage_on_the_node, packagedir (for loading the code), nprocesses, Array of directories (reading and writing results)

#rsync -av $TMPDIR/ $HOME --exclude VENV --exclude nodelist
cp $TMPDIR/*.log $HOME
cp $TMPDIR/*.dat $HOME
cp $TMPDIR/*.csv $HOME
