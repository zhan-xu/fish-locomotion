from __future__ import print_function, absolute_import
import os
import numpy as np
import argparse
import time
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from datasets import Fish_Motion
from models.motionNet.regnet import regnet

from util.logger import Logger
from util.evaluation_util import AverageMeter, accuracy_angle
from util.training_util import save_checkpoint, save_pred, adjust_learning_rate, to_numpy
from util.osutils import mkdir_p, isfile, join


idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
best_acc = 0
device = None


def main(args):
    global best_acc
    global device

    # create checkpoint dir
    mkdir_p(args.checkpoint)

    print("==> creating model regnet")
    model = regnet(color_mode=args.color_mode, num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    #criterion = torch.nn.L1Loss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'motion prediction network'
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        Fish_Motion(args.json_file, args.img_folder, args.mean_file, sigma=args.sigma, label_type=args.label_type,
                    mode=args.color_mode, reg=args.reg),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Fish_Motion(args.json_file, args.img_folder, args.mean_file, sigma=args.sigma, label_type=args.label_type,
                    train=False, mode=args.color_mode, reg=args.reg),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        print('training')
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.debug, args.flip,
                                      args.color_mode, args.reg)

        # evaluate on validation set
        print('validating')
        valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.num_classes,
                                                      args.debug, args.flip, args.color_mode, args.reg)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()


def train(train_loader, model, criterion, optimizer, debug=False, flip=True, color_mode='RGB', reg='heatmap'):
    global device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs_ori = inputs.clone()

        if color_mode != 'RGB':
            inputs = torch.mean(inputs, dim=1)
            inputs = inputs > 0.1
            inputs = inputs.float()
            inputs = inputs[:, None, :, :]
        #npimg = im_to_numpy(inputs[0,:,:,:] * 255).astype(np.uint8)
        # cv2.namedWindow('kk')
        # cv2.imshow('kk',npimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        input_var = inputs.to(device)
        target_var = target.to(device)

        # compute output
        output = model(input_var)

        score_map = output.data.cpu()
        loss = criterion(output, target_var)
        acc = accuracy_angle(score_map, target, range(9))

        print('batch: {0}, acc = {1}'.format(i, acc[0]))
        if debug:  # visualize groundtruth and predictions
            inp_d = to_numpy(inputs_ori * 255)
            inp_d = np.transpose(inp_d, (2, 3, 1, 0))
            target_d = to_numpy(target)
            tpts_d = to_numpy(meta['tpts'])[..., 0:2]
            tan_list_d = meta['tan']
            for sample in range(inp_d.shape[0]):
                img_s = inp_d[..., sample]
                pts_s = tpts_d[sample, ...].copy()
                pts_s[0:3, :] = tpts_d[sample, 7:10, :]
                pts_s[3:10, :] = tpts_d[sample, 0:7, :]
                tar_s = target_d[sample, :]
                skel_length = []
                for id_pts in range(pts_s.shape[0] - 1):
                    length = np.linalg.norm(pts_s[id_pts + 1, :] - pts_s[id_pts, :])
                    skel_length.append(length)
                pts_sec = np.zeros((pts_s.shape[0], pts_s.shape[1]))
                pts_sec[0, :] = pts_s[0, :]
                for id_pts in range(tpts_d.shape[1] - 1):
                    pts_nex = np.zeros(2)
                    pts_nex[0] = pts_s[id_pts, 0] + skel_length[id_pts] * np.cos(tar_s[id_pts] / 180.0 * np.pi)
                    pts_nex[1] = pts_s[id_pts, 1] + skel_length[id_pts] * np.sin(tar_s[id_pts] / 180.0 * np.pi)
                    pts_sec[id_pts + 1, :] = pts_nex
                print(pts_s)
                print(pts_sec)
                cv2.namedWindow('sample')
                cv2.imshow('sample', img_s)
                cv2.waitKey()
                cv2.destroyAllWindows()

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, acces.avg


def validate(val_loader, model, criterion, num_classes, debug=False, flip=True, color_mode='RGB', reg='heatmap'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes - 1)

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputs_ori = inputs.clone()

        target = target.cuda(async=True)

        if color_mode != 'RGB':
            inputs = torch.mean(inputs, dim=1)
            inputs = inputs > 0.05
            inputs = inputs.float()
            inputs = inputs[:, None, :, :]
            flip = 0

        input_var = inputs.to(device)
        target_var = target.to(device)

        # compute output
        output = model(input_var)

        score_map = output.data.cpu()
        loss = criterion(output, target_var)
        acc = accuracy_angle(score_map, target.cpu(), range(9))
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :] = score_map[n, :]
        print('batch: {0}, acc = {1}'.format(i, acc[0]))
        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, acces.avg, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')  # 2.5e-4
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 180, 240],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    parser.add_argument('--color_mode', type=str, default='L',
                        help='image color mode when processed')
    parser.add_argument('--reg', type=str, default='angle',
                        help='regression target type')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/motion_test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    # File paths
    parser.add_argument('--json_file', default='/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/motion_pred/shark_annotations_trainval_all.json',
                        type=str, help='json file path')
    parser.add_argument('--img_folder', default='/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/motion_pred',
                        type=str, help='img folder')
    parser.add_argument('--mean_file', default='/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/motion_pred/mean_bin.pth.tar',
                        type=str, help='mean binary file')

    main(parser.parse_args())
