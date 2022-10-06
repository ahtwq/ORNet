import cv2
import sys
import numpy as np
import glob
from PIL import Image
import os
from tqdm import tqdm


def process(path, scale=600):
    img = cv2.imread(path)
    x = img[int(img.shape[0]/2), :, :].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0 / r
    img_re = cv2.resize(img,(0,0),fx=s,fy=s)
    
    a = cv2.addWeighted(img_re, 4, cv2.GaussianBlur(img_re, (0,0),scale/30), -4, 128)
    b = np.zeros(a.shape, dtype=a.dtype)
    x, y = a.shape[1]//2,  a.shape[0]//2
    # 之前都是0.85
    r = int(scale*0.85)
    cv2.circle(b, (x, y), r, (255,255,255), -1, cv2.LINE_AA)
    a = cv2.bitwise_and(a, b)
    
    h, w = a.shape[:-1]
    w1 = max(x - r, 0)  
    w2 = min(x + r, w)
    h1 = max(y - r, 0)
    h2 = min(y + r, h)
    a_crop = a[h1:h2, w1:w2, :]
    img_re = cv2.bitwise_and(img_re, b)
    img_crop = img_re[h1:h2, w1:w2, :]

    # extent
    h, w = a_crop.shape[:-1]
    r_f = int(max(w, h) + max(w, h)/3)
    mask_f = np.zeros((r_f, r_f, 3), dtype=a_crop.dtype)
    we = w/6
    he = 2*w/3 - h/2
    we = int(we)
    he = int(he)
    # mask_f[he:he+h, we:we+w, :] = a_crop
    # mask_f = cv2.resize(mask_f, (256*4, 256*4))

    mask_f[he:he+h, we:we+w, :] = img_crop
    mask_f = cv2.resize(mask_f, (256*4, 256*4))
    return mask_f


folder = sys.argv[1]
save_root = sys.argv[2]
images = os.listdir(folder)
pbar = tqdm(images)
if not os.path.exists(save_root):
    os.mkdir(save_root)
for img in pbar:
    try:
        path = os.path.join(folder, img)
        img_crop = process(path)
        save_path = os.path.join(save_root, img[:-4]+'.png')
        cv2.imwrite(save_path, img_crop)
    except Exception as e:
        print(path)
        print(e)
        break
    pbar.set_description("Processing %s" % path)


