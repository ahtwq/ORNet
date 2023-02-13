import cv2
import sys
import numpy as np
import glob
from PIL import Image
import os
from tqdm import tqdm



def crop(path, scale=1000):
    img = cv2.imread(path)
    x = img[int(img.shape[0]/2), :, :].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0 /r
    img_re = cv2.resize(img,(0,0),fx=s,fy=s)

    ret, thresh = cv2.threshold(cv2.cvtColor(img_re.copy(), cv2.COLOR_BGR2GRAY) , 10, 100, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    ind = np.argmax(areas)
    (x,y), r = cv2.minEnclosingCircle(contours[ind])

    mask = np.zeros(img_re.shape, dtype=img_re.dtype)
    x = int(x)
    y = int(y)
    r = int(0.90 * r)
    cv2.circle(mask, (x, y), r, (255,255,255), -1)
    a = cv2.bitwise_and(img_re, mask)

    h, w = a.shape[:-1]
    w1 = max(x - r, 0)  
    w2 = min(x + r, w)
    h1 = max(y - r, 0)
    h2 = min(y + r, h)
    a_crop = a[h1:h2, w1:w2, :]
    
    # extent
    h, w = a_crop.shape[:-1]
    r_f = int(max(w, h) + max(w, h)/3)
    mask_f = np.zeros((r_f, r_f, 3), dtype=a_crop.dtype)
    we = w/6
    he = 2*w/3 - h/2
    we = int(we)
    he = int(he)
    mask_f[he:he+h, we:we+w, :] = a_crop
    mask_f = cv2.resize(mask_f, (1024, 1024))
    return mask_f


def crop_gray(path, scale=1000):
    img = cv2.imread(path)
    x = img[int(img.shape[0]/2), :, :].sum(1)
    r = (x>x.mean()/10).sum()/2
    s = scale * 1.0 /r
    img_re = cv2.resize(img,(0,0),fx=s,fy=s)

    rret, thresh = cv2.threshold(cv2.cvtColor(img_re.copy(), cv2.COLOR_BGR2GRAY) , 10, 100, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    ind = np.argmax(areas)
    (x,y), r = cv2.minEnclosingCircle(contours[ind])

    img_re_blur = cv2.addWeighted(img_re, 4, cv2.GaussianBlur(img_re,(0,0),scale/60), -4, 128)
    mask = np.zeros(img_re_blur.shape, dtype=img_re_blur.dtype)
    x = int(x)
    y = int(y)
    r = int(0.90 * r)
    cv2.circle(mask, (x, y), r, (255,255,255), -1)
    a = cv2.bitwise_and(img_re_blur, mask)

    h, w = a.shape[:-1]
    w1 = max(x - r, 0)  
    w2 = min(x + r, w)
    h1 = max(y - r, 0)
    h2 = min(y + r, h)
    a_crop = a[h1:h2, w1:w2, :]
    
    # extent
    h, w = a_crop.shape[:-1]
    r_f = int(max(w, h) + max(w, h)/3)
    mask_f = np.zeros((r_f, r_f, 3), dtype=a_crop.dtype)
    we = w/6
    he = 2*w/3 - h/2
    we = int(we)
    he = int(he)
    mask_f[he:he+h, we:we+w, :] = a_crop
    mask_f = cv2.resize(mask_f, (1024, 1024))
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
        img_crop = crop_gray(path)
        # f, name = path.split('/')[2:]
        # if not os.path.exists(f):
        #     os.mkdir(f)
        # save_path = os.path.join(f, name[0:-5]+'.png')
        # save_path = path.replace('predict', 'predict_scale')
        save_path = os.path.join(save_root, img[:-4]+'.png')
        cv2.imwrite(save_path, img_crop)
    except Exception as e:
        print(path)
        print(e)
    pbar.set_description("Processing %s" % path)


