#!/bin/bash


for a in {0.0,}
do
	for f in {1..9}
	do
		python nodule_3dresnet.py --lambda $a --test_fold $f --num_classes 5 --wd 1e-4 --cuda --batchSize_train 16 --LR 1e-3 --regfn sl1 --cuts -5 -4 -3 -1 --checkpoints threshold
	done
done