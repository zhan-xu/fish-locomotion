import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
import torch
import models
import datasets
from util.osutils import isfile, isdir
from util.evaluation_util import accuracy, AverageMeter
from util.vox_util import dilate_vox, three_view_with_heatmap_0, Voxcoord2Cartesian, three_view_with_heatmap
from losses import wMSELoss

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
        print("no checkpoint found")
        exit(0)

    # create model
    print("==> creating model {}".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)
    if args.wmse:
        criterion = wMSELoss().to(device)
    else:
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

    '''test_loader = torch.utils.data.DataLoader(
        datasets.Fish_Vox('/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_vox_annotations_trainval_64.json',
                          '/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data',
                          '/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64_thin',
                          sigma=args.sigma, train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)'''

    print('\nEvaluation only')
    loss, acc = test(test_loader, model, criterion, 'unet_wmse_pred.txt', args.debug, args.wmse)
    print('loss = {0}, accuracy = {1}'.format(loss, acc))
    return


def drawPred(inputs, score_map, name):
    sample_view = []
    inp = inputs[0].squeeze().numpy()
    tar = score_map[0].numpy()
    vi = three_view_with_heatmap_0(inp, tar)
    sample_view.append(vi)
    sample_view = np.concatenate(sample_view, axis=1)
    plt.imshow(sample_view)
    plt.savefig(name[0].split('.')[0]+'.png')
    plt.show()


def test(val_loader, model, criterion, res_name, debug=False, wmse=False):
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pred = {}
    for i, (inputs, target, meta) in enumerate(val_loader):
        input_var = inputs.to(device)
        target_var = target.to(device)

        # compute output
        output = model(input_var)
        score_map = output.to('cpu')
        if wmse:
            # weightes_MSE loss
            mse_weight = inputs[:,0,...].clone()
            mse_weight = mse_weight.detach()
            mse_weight = dilate_vox(mse_weight, 5)
            mse_weight_mask = mse_weight[:,np.newaxis,...]
            mse_weight_mask = np.repeat(mse_weight_mask, 10, axis=1)
            mse_weight = mse_weight.to(device)
            loss = criterion(output[0], target_var[0], mse_weight[0])
            for j in range(1, len(output)):
                loss += criterion(output[j], target_var[j], mse_weight[j])
            score_map = score_map * mse_weight_mask
        else:
            loss = criterion(output[0], target_var[0])
        for j in range(1, len(output)):
            loss += criterion(output[j], target_var[j])
        acc, pred_coord = accuracy(score_map, target_var.data.cpu())
        pred_coord = pred_coord.to('cpu').numpy()
        for k in range(pred_coord[0].shape[0]):
            pred_coord[0, k, :] = Voxcoord2Cartesian(pred_coord[0, k, :],
                                                     np.asarray(
                                                         [meta['translate'][0].numpy(), meta['translate'][1].numpy(),
                                                          meta['translate'][2].numpy()]).squeeze(),
                                                     meta['scale'][0].numpy())
        #drawPred(inputs[:,0,:,:,:], score_map, meta['name'])
        pred[meta['name'][0]] = pred_coord.tolist()[0]

        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        print("({0:d}/{1:d}) Loss: {2:.10f} | Acc: {3: .10f}".format(i + 1, len(val_loader), losses.avg, acces.avg))

        if debug:  # visualize groundtruth and predictions
            sample_view = []
            inp = inputs[0,0,...].squeeze().numpy()
            tar = target[0].numpy()
            if wmse:
                mse_weight = mse_weight.detach()
                mse_weight_np = mse_weight[0].cpu().squeeze().numpy()
                score_map_np = score_map.clone().detach()
                scr = score_map_np[0].numpy()
                vi = three_view_with_heatmap(inp, mse_weight_np, tar, scr)
            else:
                vi = three_view_with_heatmap_0(inp, tar)
            sample_view.append(vi)
            sample_view = np.concatenate(sample_view, axis=1)
            plt.imshow(sample_view)
            plt.savefig(meta['name'][0].split('.')[0] + '.png')
            plt.show()

    with open(res_name, 'w') as outfile:
        json.dump(pred, outfile)
    return losses.avg, acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 3D U-Voxel Testing')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vox_unet', choices=model_names,
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
    parser.add_argument('--sigma', type=float, default=4.0,
                        help='Groundtruth Gaussian sigma.')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/checkpoint_vox_unet', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='checkpoint/checkpoint_vox_unet/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    # experiment params
    parser.add_argument('-wmse', dest='wmse', action='store_true', help='use weighted mes loss')

    main(parser.parse_args())
