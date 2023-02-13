#!/bin/bash

# th=[-8,-5,1,5], [0.5,1.5,2.5,3.5], [0.1,0.7,1.0,2.0], [-1,0,0.5,2], [0.5,1.5,3,4.5]
for a in {0.0,}
do
	for f in {0..9}
	do
		python nodule_3dresnet.py --lambda $a --test_fold $f --num_classes 5 --wd 1e-4 --cuda --batchSize_train 16 --LR 1e-3 --regfn l1 --cuts 0.5 1.5 2.5 3.5 --checkpoints threshold
	done
done
