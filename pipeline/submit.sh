#!/bin/bash

#SBATCH --tmp=100G
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=4:00:00
#SBATCH --gpus=1


source /cluster/home/jabohl/ai4good/env/bin/activate
mkdir ${TMPDIR}/dataset
python3 main.py 
