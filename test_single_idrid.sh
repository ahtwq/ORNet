#!/bin/bash

# Only a testing
# baseline/idrid/dr
dsName='idrid'
disease='dr'
n=5
a=0
# Due to the space limit, we upload some sample images to the data folder 
# parser.add_argument('--img_root_train', type=str, default='./Data/idrid/train_resize1024', help='img_root')
# parser.add_argument('--img_root_test', type=str, default='./Data/idrid/test_resize1024', help='img_root')
trainpath='ground/idrid/example_train.txt'
valpath='ground/idrid/example_valid.txt'
testpath='ground/idrid/example_test.txt'

folder="r50_idrid_dr_base"
python train_r50_single.py --dir $folder --datasetName $dsName --disease $disease --num_classes $n --epochs 30 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --train_txtpath $trainpath --valid_txtpath $valpath --test_txtpath $testpath

# ORNet
a=0.1
folder="r50_idrid_dr_ornet"
python train_r50_single.py --dir $folder --datasetName $dsName --disease $disease --num_classes $n --epochs 30 --lr 0.0005 --batch_size 6 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --lsce --train_txtpath $trainpath --valid_txtpath $valpath --test_txtpath $testpath


