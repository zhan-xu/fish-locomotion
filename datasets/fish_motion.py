from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from util.osutils import *
from util.imutils import *
from util.transforms import *


class Fish_Motion(data.Dataset):
    def __init__(self, jsonfile, img_folder, mean_file, inp_res=256, out_res=64, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, mode='RGB', label_type='Gaussian', reg='heatmap'):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.meanstd_file = mean_file
        self.mode = mode
        self.reg = reg

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        if isfile(self.meanstd_file):
            meanstd = torch.load(self.meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, self.meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            if self.reg == 'heatmap':
                r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
                # Flip
                if random.random() <= 0.5:
                    img = torch.from_numpy(fliplr(img.numpy())).float()
                    pts[:,0] = img.size(2)-pts[:,0]
                    #pts = shufflelr(pts, width=img.size(2), dataset='mpii')
                    c[0] = img.size(2) - c[0]

            # Color
            #img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            #img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            #img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            #img.mul_(random.uniform(0.6, 1.0)).clamp_(0, 1)
        
        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        if self.reg == 'heatmap':
            target = torch.zeros(nparts, self.out_res, self.out_res)
            for i in range(nparts):
                # if tpts[i, 2] > 0: # This is evil!!
                if tpts[i, 1] > 0:
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                    target[i] = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)
        else:
            target = torch.zeros(nparts-1)
            tpts_order = tpts.clone()
            tpts_order[0:3,:] = tpts[7:10, :]
            tpts_order[3:10,:] = tpts[0:7, :]
            for i in range(nparts-1):
                tan_angle = 1.0 * (tpts_order[i+1, 1] - tpts_order[i, 1]) / (tpts_order[i+1, 0] - tpts_order[i, 0] + 1e-30)
                '''plt.plot(tpts_order.numpy()[:, 0], tpts_order.numpy()[:, 1],'ro--')
                plt.figure()
                plt.plot(pts_order.numpy()[:, 0], pts_order.numpy()[:, 1],'ro--')
                plt.show()'''
                #angle = -np.arctan(tan_angle)
                angle = np.arctan(tan_angle)
                if (tpts_order[i+1, 0] - tpts_order[i, 0])>0:
                    angle = angle / np.pi * 180
                else:
                    if angle <0:
                        angle = angle / np.pi * 180 + 180
                    else:
                        angle = angle / np.pi * 180 - 180
                # print angle
                target[i] = angle

        # Meta info
        meta = {'index' : index, 'center' : c, 'scale' : s, 
        'pts' : pts, 'tpts' : tpts}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
