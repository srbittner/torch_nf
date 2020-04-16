#!/bin/bash
source activate torch_nf

for d in 2 4 6
do
  for T in 10 25 50
  do
    for sigma in 0.5 1.0
    do
      for rs in 2 3 4 5
      do
      sbatch smcabc_mat.sh $d $T $sigma $rs
      done
    done
  done
done


