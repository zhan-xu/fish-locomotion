from __future__ import absolute_import

import os
import shutil
import torch 
import math
import numpy as np
import scipy.io


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar', snapshot=None):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds' : preds})


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def three_view_with_pointnet(model, tar, name=None):
    '''
    get top/side/front view of vox model with joint marked
    :param model: 3D point coordinate (N*3)
    :param tar: 3D joint coordinate (num_joint*3)
    '''
    model_proj1 = model[:, (1, 2)]
    tar_proj1 = tar[:,(1, 2)]
    model_proj2 = model[:, (0, 2)]
    tar_proj2 = tar[:, (0, 2)]

    model_proj3 = model[:, (0, 1)]
    tar_proj3 = tar[:, (0, 1)]

    import matplotlib.pyplot as plt
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.scatter(model_proj1[:, 0], model_proj1[:, 1], marker='o', color='blue')
    ax1.scatter(tar_proj1[:, 0], tar_proj1[:, 1], marker='o', color='red')
    ax2.scatter(model_proj2[:, 0], model_proj2[:, 1], marker='o', color='blue')
    ax2.scatter(tar_proj2[:, 0], tar_proj2[:, 1], marker='o', color='red')
    ax3.scatter(model_proj3[:, 0], model_proj3[:, 1], marker='o', color='blue')
    ax3.scatter(tar_proj3[:, 0], tar_proj3[:, 1], marker='o', color='red')
    f.subplots_adjust(hspace=0)

    if name is not None:
        plt.savefig(name[0].split('.')[0] + '.png')
    plt.show()