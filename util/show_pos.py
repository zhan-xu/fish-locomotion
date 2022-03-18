import cv2
import numpy as np
import json
import math
import os

'''img_path = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/images/0000051638.jpg'
txt_folder = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/labels'
txt_path = img_path.split('/')[-1].replace('.jpg', '.txt')
txt_path = os.path.join(txt_folder, txt_path)
txt_file = open(txt_path, 'r')
pos = json.load(txt_file)
img = cv2.imread(img_path)
print pos
txt_file.close()

img[:,:,1:] = 0
#cv2.circle(img, (int(pos['hd1'][0]), int(pos['hd1'][1])), 4, [255,255,255],2)
cv2.circle(img, (int(pos['hd2'][0]), int(pos['hd2'][1])), 4, [255,255,255],2)
cv2.circle(img, (int(pos['hd3'][0]), int(pos['hd3'][1])), 4, [255,255,255],2)
cv2.circle(img, (int(pos['sp0'][0]), int(pos['sp0'][1])), 4, [255,255,255],2)
cv2.circle(img, (int(pos['sp1'][0]), int(pos['sp1'][1])), 4, [255,255,255],2)
cv2.circle(img, (int(pos['sp2'][0]), int(pos['sp2'][1])), 4, [255,255,255],2)
cv2.circle(img, (int(pos['sp3'][0]), int(pos['sp3'][1])), 4, [255,255,255],2)

cv2.namedWindow('img')
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

'''predictions = np.load('../results/bin_res/real/preds.npy')
img_folder = '../example/real/'
for i in range(predictions.shape[0]):
    img_path = img_folder + c
    img_draw = cv2.imread(img_path)
    for p in range(predictions.shape[1]):
        cv2.circle(img_draw, (int(predictions[i, p, 0]), int(predictions[i, p, 1])), 3, (0, 0, 255), 5)
    # cv2.namedWindow('joints')
    # cv2.imshow('joints',img_draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    filename = 'res_{:03d}.png'.format(i+1)
    cv2.imwrite(filename, img_draw)'''

bin_folder = '../example/mask/'
res_folder = './'
for i in range(101):
    mask_path = bin_folder + 'mask_{:03d}.jpg'.format(i+1)
    mask = cv2.imread(mask_path)
    res_path = res_folder + 'res_{:03d}.png'.format(i+1)
    res = cv2.imread(res_path)
    img = np.concatenate((mask,res), axis=0)
    outname = 'show_{:03d}.jpg'.format(i+1)
    cv2.imwrite(outname, img)