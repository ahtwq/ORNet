import cv2
import glob
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial


def preprocess(name, img_root, tar, wrong_list):
    try:
        print(name)
        img = cv2.imread(img_root + '/' + name)
        x = img[img.shape[0] // 2, :, :].sum(1)
        thr = x.mean() / 3
        r_mask = x > thr
        left, right = np.where(r_mask==True)[0][0], np.where(r_mask==True)[0][-1]
        r = r_mask.sum() // 2
        img = img[:, left-5:right+5, :]
        mask = np.zeros(img.shape)
        cv2.circle(mask, (img.shape[1]//2, img.shape[0]//2), r, (1, 1, 1, 0), -1)
        img = (img * mask + 0 * (1 - mask)).astype(img.dtype)
        hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
        U = hsv[:, :, 2] / 255
        W = U**(1/2.2)
        V = W*255
        hsv[:, :, 2] = np.clip(V, 0, 255).astype(img.dtype)
        img = cv2.cvtColor(hsv.copy(), cv2.COLOR_HSV2BGR)
        lab = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab.copy(), cv2.COLOR_LAB2BGR)
        mask = np.zeros(img.shape)
        cv2.circle(mask, (img.shape[1]//2, img.shape[0]//2), int(r*0.9), (1, 1, 1, 0), -1)
        img = img * mask + 0 * (1 - mask)
        cv2.imwrite(tar + '/' + name[:-4]+'.png', img)
    except OSError:
        print(name)
        wrong_list.append(name)



img_root = 'trainSet0'
names = os.listdir(img_root)
save_path = 'trainSet'
try:
    os.mkdir(save_path)
except OSError:
    pass

wrong = []
print(names)
for name in tqdm(names):
    preprocess(name, img_root, save_path, wrong)

print(len(wrong))
with open('./wrong_img.txt', 'w') as file:
    file.write(str(wrong))


img_root = 'validSet0'
names = os.listdir(img_root)
save_path = 'validSet'
try:
    os.mkdir(save_path)
except OSError:
    pass

wrong = []
print(names)
for name in tqdm(names):
    preprocess(name, img_root, save_path, wrong)

print(len(wrong))
with open('./wrong_img.txt', 'w') as file:
    file.write(str(wrong))


img_root = 'testSet0'
names = os.listdir(img_root)
save_path = 'testSet'
try:
    os.mkdir(save_path)
except OSError:
    pass

wrong = []
print(names)
for name in tqdm(names):
    preprocess(name, img_root, save_path, wrong)

print(len(wrong))
with open('./wrong_img.txt', 'w') as file:
    file.write(str(wrong))