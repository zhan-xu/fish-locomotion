from __future__ import print_function, absolute_import

import numpy as np
import json
import random
import util.binvox_rw as binvox_rw

import torch
import torch.utils.data as data
import torch.nn.functional as F

from util.osutils import *
from util.vox_util import draw_labelmap


def draw2Dview(view, coord1, coord2, radius=1):
    for p in range(len(coord1)):
        ll = max(coord1[p] - radius, 0)
        rr = min(coord1[p] + radius, view.shape[0])
        tt = max(coord2[p] - radius, 0)
        bb = min(coord2[p] + radius, view.shape[1])
        view[ll:rr,tt:bb,:] = (255, 0, 0)
    return view


def three_view_with_joint(model, pts, radius=1):
    '''
    get top/side/front view of vox model with joint marked
    :param model: 3D vox array
    :param pts: numpy array with vox coordinates of joints
    :param radius: radius for each joint mark
    :return: top/side/front view of the model
    '''
    x = pts[:,0].numpy().astype(np.int8)
    y = pts[:,1].numpy().astype(np.int8)
    z = pts[:,2].numpy().astype(np.int8)
    view1 = np.sum(model, axis=0)
    view1 = (np.repeat(view1[..., np.newaxis], 3, axis=2) > 0) * np.ones((view1.shape[0], view1.shape[1], 3), dtype=np.uint8) * 255
    view1 = draw2Dview(view1, y, z, radius)
    view2 = np.sum(model, axis=1)
    view2 = (np.repeat(view2[..., np.newaxis], 3, axis=2) > 0) * np.ones((view2.shape[0], view2.shape[1], 3), dtype=np.uint8) * 255
    view2 = draw2Dview(view2, x, z, radius)
    view3 = np.sum(model, axis=2)
    view3 = (np.repeat(view3[..., np.newaxis], 3, axis=2) > 0) * np.ones((view3.shape[0], view3.shape[1], 3), dtype=np.uint8) * 255
    view3 = draw2Dview(view3, x, y, radius)
    view = np.concatenate((view1, view2, view3))

    plt.imshow(view)
    plt.show()


class Fish_Vox(data.Dataset):
    def __init__(self, jsonfile, img_folder, thin_folder, inp_res=64, out_res=68, train=True, sigma=1):
        self.img_folder = img_folder    # root image folders
        self.thin_folder = thin_folder  # thin skeleton folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
            #self.valid.append(idx)
        # to reduce I/O, for fish dataset, we read in all data once
        '''self.all_models = {}
        if self.is_train:
            for idx, val in enumerate(self.anno):
                if not val['isValidation']:
                    model_path = os.path.join(self.img_folder, val['img_paths'])
                    with open(model_path, 'rb') as f:
                        model = binvox_rw.read_as_3d_array(f).data
                    thin_model_path = os.path.join(self.thin_folder, model_path.split('/')[-1])
                    with open(thin_model_path, 'rb') as f:
                        model_thin = binvox_rw.read_as_3d_array(f).data
                    self.all_models[idx] = (model, model_thin)
        else:
            for idx, val in enumerate(self.anno):
                if val['isValidation']:
                    model_path = os.path.join(self.img_folder, val['img_paths'])
                    with open(model_path, 'rb') as f:
                        model = binvox_rw.read_as_3d_array(f).data
                    thin_model_path = os.path.join(self.thin_folder, model_path.split('/')[-1])
                    with open(thin_model_path, 'rb') as f:
                        model_thin = binvox_rw.read_as_3d_array(f).data
                    self.all_models[idx] = (model, model_thin)'''

    def __getitem__(self, index):
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        pts = torch.Tensor(a['joint'])
        t = a['translate']
        s = a['scale']

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        model_path = os.path.join(self.img_folder, a['img_paths'])
        with open(model_path, 'rb') as f:
            model = binvox_rw.read_as_3d_array(f).data.astype(float)
        thin_model_path = os.path.join(self.thin_folder, model_path.split('/')[-1])
        with open(thin_model_path, 'rb') as f:
            model_thin = binvox_rw.read_as_3d_array(f).data.astype(float)
        #model = self.all_models[index][0].astype(float)
        #model_thin = self.all_models[index][1].astype(float)

        # mirror
        if self.is_train:
            mirror_dic = random.random()
            if mirror_dic <= 0.25:
                model = np.flip(model, axis=0).copy()
                model_thin = np.flip(model, axis=0).copy()
                pts[:, 0] = a['dims'][0] - pts[:, 0] - 1
            elif mirror_dic>0.25 and mirror_dic<=0.5:
                model = np.flip(model, axis=1).copy()
                model_thin = np.flip(model, axis=1).copy()
                pts[:, 1] = a['dims'][1] - pts[:, 1] - 1
            elif mirror_dic>0.5 and mirror_dic<=0.75:
                model = np.flip(model, axis=2).copy()
                model_thin = np.flip(model, axis=2).copy()
                pts[:, 2] = a['dims'][2] - pts[:, 2] - 1

        #three_view_with_joint(model,pts)
        model = model[..., np.newaxis]
        model_thin = model_thin[...,np.newaxis]
        model = np.transpose(model, (3, 0, 1, 2))  # C*H*W
        model_thin = np.transpose(model_thin, (3, 0, 1, 2))
        model = np.concatenate((model,model_thin), axis=0)
        model = torch.from_numpy(model)
        padding_size = 2
        model = F.pad(model, (padding_size, padding_size, padding_size, padding_size, padding_size,padding_size), "constant", 0)
        model = model.float()
        if model.max() > 1:
            model /= 255

        # Generate ground truth
        target = torch.zeros(nparts, self.out_res, self.out_res, self.out_res)
        for i in range(nparts):
            target[i] = draw_labelmap(target[i], pts[i]+padding_size, self.sigma)

        # Meta info
        meta = {'index': index, 'translate': t, 'scale': s, 'pts': pts, 'name': model_path.split('/')[-1]}

        return model, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
