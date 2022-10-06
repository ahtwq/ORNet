#!/bin/bash


for a in {0,}
do
	for f in {0..9}
	do
		python nodule_3dresnet_olr.py --lambda $a --test_fold $f --num_classes 5 --wd 1e-4  --cuda --batchSize_train 16 --LR 5e-3 --cuts 0.5 1.5 2.5 3.5 --threshold learnable
	done
done
