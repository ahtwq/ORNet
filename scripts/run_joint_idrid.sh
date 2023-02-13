#!/bin/bash


## Baseline
dsName='idrid'
n1=5
n2=3
a=0
# folder="r50_base_joint"
# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt sgd
# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adamw

## ORNet
a=0.3
folder="r50_base_joint"
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt sgd
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adamw

## ORNet
a=0.5
folder="r50_base_joint"
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt sgd
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adamw

## ORNet
a=0.7
folder="r50_base_joint"
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt sgd
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adam
python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --opt adamw

## CANet
# folder="canetr50_base_joint"
# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --bname canet --opt adamw --canet_lambda 0.1

# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --bname canet --opt adamw --canet_lambda 0.25

# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --bname canet --opt adamw --canet_lambda 0.5

# python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.0001 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --bname canet --opt adamw --canet_lambda 0.7