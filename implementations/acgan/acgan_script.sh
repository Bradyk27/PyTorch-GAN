#!/bin/bash

#SBATCH -N 4
#SBATCH --job-name=acgan
#SBATCH --output=acgan.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bradyakruse@gmail.com

enable_lmod
module load container_env tensorflow-gpu/2.2.0
crun.tensorflow-gpu python acgan.py
