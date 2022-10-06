#!/bin/bash


# messidor, Baseline
# 10 cross validation
tra_root='./Data/messidor/messidor_resize512'
val_root='./Data/messidor/messidor_resize512'
dsName='messidor'
disease='dr'
n=4
a=0

for i in {0..9}
do
    tra_path="ground/messidor/train${i}.txt"
    val_path="ground/messidor/valid${i}.txt"
    test_path="ground/messidor/test${i}.txt"
    folder="dr/dr_${i}"
    python train_r50_single.py --dir $folder --datasetName $dsName --disease $disease --num_classes $n --epochs 180 --lr 0.001 --batch_size 16 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --img_root_train $tra_root --img_root_test $val_root --train_txtpath $tra_path --valid_txtpath $val_path --test_txtpath $test_path
done


# disease='dme'
# n=3
# a=0
# for i in {0..9}
# do
#     tra_path="ground/messidor/train${i}.txt"
#     val_path="ground/messidor/valid${i}.txt"
#     test_path="ground/messidor/test${i}.txt"
#     folder="dme/dme_${i}"
#     python train_r50_single.py --dir $folder --datasetName $dsName --disease $disease --num_classes $n --epochs 180 --lr 0.001 --batch_size 16 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --img_root_train $tra_root --img_root_test $val_root --train_txtpath $tra_path --valid_txtpath $val_path --test_txtpath $test_path
# done