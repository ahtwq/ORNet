#!/bin/bash


dsName='messidor'
n1=4
n2=3
a=0
folder="r50_base_joint"
tra_root='./Data/messidor/messidor_resize512'
val_root='./Data/messidor/messidor_resize512'
# 10 cross validation; Baseline
a=0.0
for i in {0..9}
do
    tra_path="ground/messidor/train${i}.txt"
    val_path="ground/messidor/valid${i}.txt"
    test_path="ground/messidor/test${i}.txt"
    python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 180 --lr 0.001 --batch_size 16 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --img_root_train $tra_root --img_root_test $val_root --train_txtpath $tra_path --valid_txtpath $val_path --test_txtpath $test_path
done


# 10 cross validation; ORNet 
# a=0.1
# for i in {0..9}
# do
#     tra_path="ground/messidor/train${i}.txt"
#     val_path="ground/messidor/valid${i}.txt"
#     test_path="ground/messidor/test${i}.txt"
#     python train_r50_joint.py --dir $folder --datasetName $dsName --num_classes1 $n1 --num_classes2 $n2 --epochs 16 --lr 0.001 --batch_size 16 --momentum 0.95 --wd 1e-4 --gamma 0.5 --nesterov --alpha $a --img_root_train $tra_root --img_root_test $val_root --train_txtpath $tra_path --valid_txtpath $val_path --test_txtpath $test_path
# done
