import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from collections import Counter


def split_train_valtest(file, nfold=5):
    with open(file, 'r') as f:
        fls = f.readlines()
    fls = [line.strip() for line in fls]
    ys = [int(line.split()[1]) for line in fls]
    print('Total lines:', len(np.array(ys)))

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=1)
    i = 0
    for train_indx, val_test_indx in skf.split(ys, ys):
        train_list = [fls[i] for i in train_indx]
        val_test_list = [fls[i] for i in val_test_indx]

        f1 = open('train{}.txt'.format(i), 'w')
        f2 = open('valid_test{}.txt'.format(i), 'w')
        print(len(train_indx), len(val_test_indx))
        for item in train_list:
            f1.write(item + '\n')

        labels = []
        for item in val_test_list:
            f2.write(item + '\n')
            labels.append(int(item.split()[-1]))
        labels = sorted(labels)
        print(Counter(labels))
        i += 1
        print('-'*50)


def split_val_test(txtfile, save_txts=['train.txt', 'valid.txt'], test_size=0.1):
    with open(txtfile, 'r') as f:
        fls = f.readlines()
    fls = [line.strip() for line in fls]
    ys = [line.split(',')[-2]+line.split(',')[-1] for line in fls]
    print(len(np.array(ys)))

    skf = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1)
    for val_indx, test_indx in skf.split(ys, ys):
        val_list = [fls[i] for i in val_indx]
        test_list = [fls[i] for i in test_indx]
        f1 = open(save_txts[0], 'w')
        f2 = open(save_txts[1], 'w')
        print(len(val_indx), len(test_indx))
        for item in val_list:
            f1.write(item + '\n')
        for item in test_list:
            f2.write(item + '\n')


# Main
# file = 'dset/allData.txt'
# nfold = 5
# split_train_valtest(file, nfold)

txtfile = 'train_valid.txt'
save_txts = ['train_0.2.txt', 'valid_0.2.txt']
test_size = 0.2
split_val_test(txtfile, save_txts, test_size)

