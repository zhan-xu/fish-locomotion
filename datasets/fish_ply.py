from __future__ import print_function, absolute_import

import numpy as np
import json
import random
import cv2

import torch
import torch.utils.data as data

from util.osutils import *
from util.mesh_obj import Mesh_obj


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

    cv2.namedWindow('view')
    cv2.imshow('view', view)
    cv2.waitKey()
    cv2.destroyAllWindows()


class Fish_Ply(data.Dataset):
    def __init__(self, jsonfile, img_folder, sample_num, train=True):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.sample_num = sample_num

        # create train/val split
        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation']:
                self.valid.append(idx)
            else:
                self.train.append(idx)

    def __getitem__(self, index):
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        pts = torch.Tensor(a['joint'])

        # For single-person pose estimation with a centered/scaled figure
        model_path = os.path.join(self.img_folder, a['img_paths'])
        model = Mesh_obj(model_path).v
        if len(model) < self.sample_num:
            model = np.vstack((model, model[-1]))

        # mirror
        if self.is_train:
            mirror_dic = random.random()
            if mirror_dic <= 0.25:
                model[:, 0] = -model[:,0]
                pts[:, 0] = -pts[:, 0]
            elif mirror_dic>0.25 and mirror_dic<=0.5:
                model[:, 1] = -model[:, 1]
                pts[:, 1] = -pts[:, 1]
            elif mirror_dic>0.5 and mirror_dic<=0.75:
                model[:, 2] = -model[:, 2]
                pts[:, 2] = -pts[:, 2]

        model = np.float32(model)
        model = torch.from_numpy(model)
        target = pts

        # Meta info
        meta = {'index': index, 'name': model_path.split('/')[-1]}

        return model, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
