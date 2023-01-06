#!/bin/bash

#SBATCH --tmp=100G
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=1
#SBATCH --time=4:00:00


source PATH_TO_ENV/bin/activate
mkdir ${TMPDIR}/reforestree
mkdir ${TMPDIR}/dataset
rsync -raq PATH_TO_REFORESTREE/* ${TMPDIR}/reforestree/.
python3 main.py 
