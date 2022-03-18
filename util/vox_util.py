from __future__ import absolute_import
from .training_util import *
import scipy.ndimage as ndimage
import torch.nn.functional as F


def draw_labelmap(img, pt, sigma):
    # Draw a 3D gaussian
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma), int(pt[2] - 3*sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1), int(pt[2] + 3*sigma +1)]
    if (ul[0] >= img.shape[0] or ul[1] >= img.shape[1] or ul[2] >= img.shape[2] or
            br[0] < 0 or br[1] < 0 or br[2] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    z = y[..., np.newaxis]
    x0 = y0 = z0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[0]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[1]) - ul[1]
    g_z = max(0, -ul[2]), min(br[2], img.shape[2]) - ul[2]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[0])
    img_y = max(0, ul[1]), min(br[1], img.shape[1])
    img_z = max(0, ul[2]), min(br[2], img.shape[2])

    img[img_x[0]:img_x[1], img_y[0]:img_y[1], img_z[0]:img_z[1]] = g[g_x[0]:g_x[1], g_y[0]:g_y[1], g_z[0]:g_z[1]]
    return to_torch(img)


def dilate_vox(vox_in, r=3):
    vox_in = F.threshold(vox_in, 0.01, 0)
    vox_in = (vox_in > 0.0).float()
    vox_np = vox_in.numpy()
    struct = np.ones((2*r+1, 2*r+1, 2*r+1)).astype(bool)
    for i in range(vox_np.shape[0]):
        vox_np[i,...] = ndimage.binary_dilation(vox_np[i,...].squeeze(), structure=struct)
    vox_out = torch.from_numpy(vox_np)
    return vox_out

def Cartesian2Voxcoord(v, translate, scale, resolution=(64, 64, 64)):
    vc = (v - translate) / scale * resolution
    vc = vc.astype(int)
    return vc[0], vc[1], vc[2]


def Voxcoord2Cartesian(vc, translate, scale, resolution=(64, 64, 64)):
    v = vc / resolution * scale + translate
    return v[0], v[1], v[2]


def three_view_with_heatmap_0(model, heatmap):
    view1 = []
    hm = np.sum(model, axis=0)
    hm = (hm > 0) * np.ones(hm.shape, dtype=np.uint8) * 255
    hm = hm[..., np.newaxis]
    hm_0 = np.concatenate((hm, np.zeros(hm.shape, dtype=np.uint8), np.zeros(hm.shape, dtype=np.uint8)), axis=2)
    view1.append(hm_0)
    for j in range(heatmap.shape[0]):
        hm = heatmap[j, ...]
        hm = np.sum(hm, axis=0)
        hm = hm * (255.0 / np.max(hm))
        hm = hm.astype(np.uint8)
        hm_new = hm_0.copy()
        hm_new[:, :, 1] = hm
        view1.append(hm_new)
    view1 = np.concatenate(view1)

    view2 = []
    hm = np.sum(model, axis=1)
    hm = (hm > 0) * np.ones(hm.shape, dtype=np.uint8) * 255
    hm = hm[..., np.newaxis]
    hm_0 = np.concatenate((hm, np.zeros(hm.shape, dtype=np.uint8), np.zeros(hm.shape, dtype=np.uint8)), axis=2)
    view2.append(hm_0)
    for j in range(heatmap.shape[0]):
        hm = heatmap[j, ...]
        hm = np.sum(hm, axis=1)
        hm = hm * (255.0 / np.max(hm))
        hm = hm.astype(np.uint8)
        hm_new = hm_0.copy()
        hm_new[:, :, 1] = hm
        view2.append(hm_new)
    view2 = np.concatenate(view2)

    view3 = []
    hm = np.sum(model, axis=2)
    hm = (hm > 0) * np.ones(hm.shape, dtype=np.uint8) * 255
    hm = hm[..., np.newaxis]
    hm_0 = np.concatenate((hm, np.zeros(hm.shape, dtype=np.uint8), np.zeros(hm.shape, dtype=np.uint8)), axis=2)
    view3.append(hm_0)
    for j in range(heatmap.shape[0]):
        hm = heatmap[j, ...]
        hm = np.sum(hm, axis=2)
        hm = hm * (255.0 / np.max(hm))
        hm = hm.astype(np.uint8)
        hm_new = hm_0.copy()
        hm_new[:, :, 1] = hm
        view3.append(hm_new)
    view3 = np.concatenate(view3)

    view = np.concatenate((view1, view2, view3), axis=1)
    return view


def three_view_with_heatmap(model, wmse, heatmap, scoremap):
    '''
    get top/side/front view of vox model with joint marked
    :param model: 3D vox array
    :param heatmap: numpy array of all heatmap
    :param scoremap: predicted score maps
    :return: top/side/front view of the model
    '''
    view1 = []
    hm = np.sum(model, axis=0)
    hm = (hm > 0) * np.ones(hm.shape, dtype=np.uint8) * 255
    hm = hm[...,np.newaxis]
    hm_0 = np.concatenate((hm, np.zeros(hm.shape,dtype=np.uint8), np.zeros(hm.shape,dtype=np.uint8)), axis=2)
    view1.append(hm_0)
    for j in range(heatmap.shape[0]):
        hm = heatmap[j,...]
        hm = np.sum(hm, axis=0)
        hm = hm * (255.0/np.max(hm))
        hm = hm.astype(np.uint8)
        hm_new = hm_0.copy()
        hm_new[:,:,1] = hm
        view1.append(hm_new)
    view1 = np.concatenate(view1)

    view1_scr = []
    w_flat = np.sum(wmse, axis=0)
    w_flat = (w_flat > 0) * np.ones(w_flat.shape, dtype=np.uint8) * 255
    w_flat = w_flat[...,np.newaxis]
    w_flat_0 = np.concatenate((w_flat, np.zeros(w_flat.shape, dtype=np.uint8), np.zeros(w_flat.shape, dtype=np.uint8)), axis=2)
    view1_scr.append(w_flat_0)
    for j in range(scoremap.shape[0]):
        scr = scoremap[j, ...]
        scr = np.sum(scr, axis=0)
        scr = scr * (255.0 / np.max(scr))
        scr = scr.astype(np.uint8)
        scr_new = hm_0.copy()
        scr_new[:, :, 1] = scr
        view1_scr.append(scr_new)
    view1_scr = np.concatenate(view1_scr)

    view1 = np.concatenate((view1, view1_scr), axis=1)

    view2 = []
    hm = np.sum(model, axis=1)
    hm = (hm > 0) * np.ones(hm.shape, dtype=np.uint8) * 255
    hm = hm[..., np.newaxis]
    hm_0 = np.concatenate((hm, np.zeros(hm.shape, dtype=np.uint8), np.zeros(hm.shape, dtype=np.uint8)), axis=2)
    view2.append(hm_0)
    for j in range(heatmap.shape[0]):
        hm = heatmap[j, ...]
        hm = np.sum(hm, axis=1)
        hm = hm * (255.0/np.max(hm))
        hm = hm.astype(np.uint8)
        hm_new = hm_0.copy()
        hm_new[:, :, 1] = hm
        view2.append(hm_new)
    view2 = np.concatenate(view2)

    view2_scr = []
    w_flat = np.sum(wmse, axis=1)
    w_flat = (w_flat > 0) * np.ones(w_flat.shape, dtype=np.uint8) * 255
    w_flat = w_flat[..., np.newaxis]
    w_flat_0 = np.concatenate((w_flat, np.zeros(w_flat.shape, dtype=np.uint8), np.zeros(w_flat.shape, dtype=np.uint8)), axis=2)
    view2_scr.append(w_flat_0)
    for j in range(scoremap.shape[0]):
        scr = scoremap[j, ...]
        scr = np.sum(scr, axis=1)
        scr = scr * (255.0 / np.max(scr))
        scr = scr.astype(np.uint8)
        scr_new = hm_0.copy()
        scr_new[:, :, 1] = scr
        view2_scr.append(scr_new)
    view2_scr = np.concatenate(view2_scr)
    view2 = np.concatenate((view2, view2_scr), axis=1)

    view3 = []
    hm = np.sum(model, axis=2)
    hm = (hm > 0) * np.ones(hm.shape, dtype=np.uint8) * 255
    hm = hm[..., np.newaxis]
    hm_0 = np.concatenate((hm, np.zeros(hm.shape, dtype=np.uint8), np.zeros(hm.shape, dtype=np.uint8)), axis=2)
    view3.append(hm_0)
    for j in range(heatmap.shape[0]):
        hm = heatmap[j, ...]
        hm = np.sum(hm, axis=2)
        hm = hm * (255.0/np.max(hm))
        hm = hm.astype(np.uint8)
        hm_new = hm_0.copy()
        hm_new[:, :, 1] = hm
        view3.append(hm_new)
    view3 = np.concatenate(view3)

    view3_scr = []
    w_flat = np.sum(wmse, axis=2)
    w_flat = (w_flat > 0) * np.ones(w_flat.shape, dtype=np.uint8) * 255
    w_flat = w_flat[..., np.newaxis]
    w_flat_0 = np.concatenate((w_flat, np.zeros(w_flat.shape, dtype=np.uint8), np.zeros(w_flat.shape, dtype=np.uint8)), axis=2)
    view3_scr.append(w_flat_0)
    for j in range(scoremap.shape[0]):
        scr = scoremap[j, ...]
        scr = np.sum(scr, axis=2)
        scr = scr * (255.0 / np.max(scr))
        scr = scr.astype(np.uint8)
        scr_new = hm_0.copy()
        scr_new[:, :, 1] = scr
        view3_scr.append(scr_new)
    view3_scr = np.concatenate(view3_scr)
    view3 = np.concatenate((view3, view3_scr), axis=1)

    view = np.concatenate((view1, view2, view3), axis=1)
    return view