#!/bin/bash


dsName='idrid'
n1=5
n2=3
a=0

# CANet
folder="canetr50_base_joint"
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --bname canet --canet_lambda 0.25