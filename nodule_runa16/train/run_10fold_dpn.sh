#!/bin/bash

a=0
for f in {0..9}
do
	python nodule_dpn.py --lambda $a --test_fold $f --num_classes 5 --wd 1e-4  --cuda --batchSize_train 8 --LR 1e-3
done

