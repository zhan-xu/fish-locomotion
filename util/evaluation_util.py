from __future__ import absolute_import
from .training_util import *

__all__ = ['accuracy', 'AverageMeter']

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 5, 'Score maps should be 5-dim'
    assert scores.size(2) == scores.size(3) == scores.size(4), 'Score should have equal length on each dimension'
    score_dim = scores.size(4)
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 3).float()

    preds[:,:,0] = torch.floor(preds[:,:,0] / (score_dim*score_dim)) - 1
    preds[:,:,1] = torch.floor(preds[:,:,1] / score_dim) % score_dim - 1
    preds[:,:,2] = preds[:,:,2] % score_dim - 1

    pred_mask = maxval.gt(0).repeat(1, 1, 3).float()
    preds *= pred_mask
    return preds

def get_preds_motion(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = (preds[:,:,0] - 1) % scores.size(3) + 1
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=2):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1


def accuracy(output, target):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds = get_preds(output)
    gts = get_preds(target)
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
                dists[c, n] = torch.dist(preds[n, c, :], gts[n, c, :])

    acc = torch.zeros(preds.size(1)+1)
    avg_acc = 0
    cnt = 0

    for i in range(preds.size(1)):
        acc[i+1] = dist_acc(dists[i], thr=3)
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc, preds


def accuracy_angle(output, target, idxs, thr=5):
    acc = torch.zeros(len(idxs) + 1)
    acc[1:] = (output - target).abs().le(thr).sum(dim=0).float() / output.size(0)
    acc[0] = acc[1:].sum() / len(idxs)
    return acc


def dist_acc_coord(output, target, thr=2):
    if target.ndim == 1:
        diff = np.linalg.norm(output - target, ord=None)
        return 1.0 * (diff < thr)
    else:
        diff = np.linalg.norm(output - target, ord=None, axis=1)
        return 1.0 * np.sum(diff<thr) / len(diff)


def accuracy_coord(output, target):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    out_np = output.numpy()
    tar_np = target.numpy()
    out_np = np.resize(out_np,(out_np.shape[0],10, 3))
    tar_np = np.resize(tar_np,(tar_np.shape[0],10, 3))

    acc = torch.zeros(tar_np.shape[1] + 1)
    avg_acc = 0
    cnt = 0

    for i in range(tar_np.shape[1]):
        acc[i + 1] = dist_acc_coord(out_np[:,i,:].squeeze(), tar_np[:,i,:].squeeze(), thr=2e-2)
        avg_acc = avg_acc + acc[i + 1]
        cnt += 1
    acc[0] = avg_acc / cnt

    return acc


def final_preds(output, res):
    coords = get_preds(output) # float type
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
