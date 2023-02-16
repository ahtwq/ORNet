#!/bin/bash


# baseline/idrid/dr
dsName='idrid'
disease='dr'
n=5
a=0
folder="r50_idrid_dr_base"
python train_r50_single.py --dir $folder --datasetName $dsName --disease $disease --num_classes $n --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce

# ORNet
a=0.1
folder="r50_idrid_dr_ornet"
python train_r50_single.py --dir $folder --datasetName $dsName --disease $disease -num_classes $n --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce


