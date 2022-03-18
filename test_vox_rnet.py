import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import json
import models
import datasets
from util.osutils import isfile, isdir
from util.vox_util import draw_labelmap, three_view_with_heatmap_0, Voxcoord2Cartesian
from util.evaluation_util import accuracy_coord, AverageMeter

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
    print("==> creating model {}".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = torch.nn.MSELoss(size_average=True).to(device)

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
    test_loader = torch.utils.data.DataLoader(
        datasets.Fish_Vox('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_vox_annotations_test.json',
                          '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data',
                          '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test_thin',
                          sigma=args.sigma, train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print('\nTest only')
    loss, acc = test(test_loader, model, criterion, 'rnet_pred.txt', args.debug)
    print('loss = {0}, accuracy = {1}'.format(loss, acc))
    return


def drawPred(inputs, pts, name):
    pts = pts.squeeze()
    target = torch.zeros(10, 68, 68, 68)
    for i in range(10):
        target[i] = draw_labelmap(target[i], pts[i] + 2, 2)

    sample_view = []
    inp = inputs[0].squeeze().numpy()
    tar = target.numpy()
    vi = three_view_with_heatmap_0(inp, tar)
    sample_view.append(vi)
    sample_view = np.concatenate(sample_view, axis=1)
    plt.imshow(sample_view)
    plt.savefig(name[0].split('.')[0]+'.png')
    plt.show()
    #cv2.imwrite(name[0].split('.')[0]+'.png', sample_view)
    return target


def test(val_loader, model, criterion, res_name, debug=False):
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pred = {}
    for i, (inputs, target, meta) in enumerate(val_loader):
        if debug:  # visualize groundtruth and predictions
            sample_view = []
            for i in range(min(4, inputs.size(0))):
                inp = inputs[i].squeeze().numpy()
                tar = target[i].numpy()
                vi = three_view_with_heatmap_0(inp, tar)
                sample_view.append(vi)
            sample_view = np.concatenate(sample_view, axis=1)
            plt.imshow(sample_view)
            plt.show()

        input_var = inputs.to(device)

        target_var = meta['pts'].view(meta['pts'].shape[0], -1).to(device)

        # compute output
        output = model(input_var)
        score_map = output.data.cpu()

        loss = criterion(output[0], target_var[0])
        for j in range(1, len(output)):
            loss += criterion(output[j], target_var[j])
        acc = accuracy_coord(score_map, target_var.data.cpu())
        pred_coord = np.resize(score_map, (score_map.shape[0], 10, 3))
        drawPred(inputs[:,0,:,:,:], pred_coord, meta['name'])
        for i in range(pred_coord[0].shape[0]):
            pred_coord[0, i, :] = Voxcoord2Cartesian(pred_coord[0, i, :],
                                                     np.asarray(
                                                         [meta['translate'][0].numpy(), meta['translate'][1].numpy(),
                                                          meta['translate'][2].numpy()]).squeeze(),
                                                     meta['scale'][0].numpy())
        pred[meta['name'][0]] = pred_coord.tolist()[0]

        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        print("({0:d}/{1:d}) Loss: {2:.10f} | Acc: {3: .10f}".format(i + 1, len(val_loader), losses.avg, acces.avg))

    with open(res_name, 'w') as outfile:
        json.dump(pred, outfile)

    return losses.avg, acces.avg


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
    parser.add_argument('-c', '--checkpoint', default='checkpoint/checkpoint_vox_rnet', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='checkpoint/checkpoint_vox_rnet/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    main(parser.parse_args())
