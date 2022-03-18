import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import json
import glob
import models
import datasets
from plyfile import PlyData
from util.osutils import isfile, isdir
from util.vox_util import draw_labelmap
from util.evaluation_util import accuracy_coord, AverageMeter
from util.vox_util import three_view_with_heatmap_0, Voxcoord2Cartesian
from models.pointNet.pointnet import PointNetReg as Pointnet
from open3d import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_acc = 0
device = None


def main(args):
    global best_acc
    global device

    # create checkpoint dir
    if not isdir(args.checkpoint):
        print("Specified folder is not found.")
        exit(0)

    # create model
    print("==> creating model point_net")
    model = Pointnet(k=3 * args.num_classes, num_points=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = torch.nn.MSELoss(reduction='elementwise_mean').to(device)

    # optionally resume from a checkpoint
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit(0)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data loading code
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.Fish_Ply('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_ply_annotations_test_0913.json',
    #                       '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data', sample_num=1000, train=False),
    #     batch_size=args.test_batch, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    print('\nTest only')
    test(model, args.data_folder, 'test_3d_model/pointnet_pred.txt')
    return


def drawCube(center, radius, color=[0.0,0.0,0.0]):
    mesh_sphere = create_mesh_sphere(radius=radius)
    transform_mat = np.eye(4)
    transform_mat[0:3, -1] = center
    mesh_sphere.transform(transform_mat)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere


def test(model, input_folder, outfile, vis=True):
    # switch to evaluate mode
    model.eval()
    ply_list = glob.glob(input_folder+'*.ply')
    for ply_file in ply_list:
        print(ply_file)
        plydata = PlyData.read(ply_file)
        num_sample = len(plydata.elements[0].data)
        inputs = np.zeros((1, num_sample, 3), dtype=np.float32)
        inputs[0, :, 0] = plydata['vertex']['x']
        inputs[0, :, 1] = plydata['vertex']['y']
        inputs[0, :, 2] = plydata['vertex']['z']
        if inputs.shape[1]< 1000:
            inputs = np.concatenate((inputs, inputs[:,-(1000-inputs.shape[1]):,:]), axis=1)
        inputs = torch.from_numpy(inputs)
        inputs = np.transpose(inputs, (0, 2, 1))
        input_var = inputs.to(device)

        # compute output
        output,_ = model(input_var)
        score_map = output.data.cpu()
        pred_coord = np.resize(score_map, (score_map.shape[0], 10, 3))
        pred = pred_coord.squeeze()
        np.savetxt(ply_file.replace('.ply', '.txt'), pred)
        if vis:
            vis = Visualizer()
            vis.create_window()
            pcd = PointCloud()
            pcd.points = Vector3dVector(inputs.numpy().squeeze().transpose())
            vis.add_geometry(pcd)
            for joint_pos in pred:
                vis.add_geometry(drawCube(joint_pos, 0.007, color=[1.0, 0.0, 0.0]))
            vis.run()
            vis.destroy_window()

        print('Done.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 3D R-Voxel Testing')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vox_rnet', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vox_unet)')
    parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('--workers', '-j', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--test-batch', default=1, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Groundtruth Gaussian sigma.')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/pointnet_classic', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='checkpoint/pointnet_classic/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--data_folder', default='./test_3d_model/', type=str, metavar='PATH',
                        help='path to test data)')

    main(parser.parse_args())
