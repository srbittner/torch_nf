#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=smcabc_mat
#SBATCH -c 1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1gb

source activate torch_nf
python3 smcabc_mat.py --d $1 --T $2 --sigma $3 --rs $4
