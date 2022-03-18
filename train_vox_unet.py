import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import models
import datasets
from util.osutils import mkdir_p, isfile, isdir, join
from util.logger import Logger
from util.training_util import adjust_learning_rate, save_checkpoint
from util.evaluation_util import accuracy, AverageMeter
from util.vox_util import dilate_vox, three_view_with_heatmap
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
        print("build new folder")
    mkdir_p(args.checkpoint)

    # create model
    print("==> creating model {}".format(args.arch))
    model = models.__dict__[args.arch](num_classes=args.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    if args.wmse:
        criterion = wMSELoss().to(device)
    else:
        criterion = torch.nn.MSELoss(size_average=True).to(device)


    '''optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)'''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = args.arch
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

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        datasets.Fish_Vox('data/fish_vox_annotations_trainval_64.json',
                          'data',
                          'data/voxel/trainval_64_thin',
                          sigma=args.sigma),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.Fish_Vox('data/fish_vox_annotations_trainval_64.json',
                          'data',
                          'data/voxel/trainval_64_thin',
                          sigma=args.sigma, train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    '''train_loader = torch.utils.data.DataLoader(
        datasets.Fish_Vox('/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_vox_annotations_trainval_64.json',
                          '/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data',
                          '/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64_thin',
                          sigma=args.sigma),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.Fish_Vox('/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_vox_annotations_trainval_64.json',
                          '/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data',
                          '/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64_thin',
                          sigma=args.sigma, train=False),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)'''

    if args.evaluate:
        print('\nEvaluation only')
        loss, acc = validate(val_loader, model, criterion, args.debug)
        print('loss = {0}, accuracy = {1}'.format(loss, acc))
        return

    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.debug, args.wmse)

        # evaluate on validation set
        valid_loss, valid_acc = validate(val_loader, model, criterion, args.debug, args.wmse)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot(['Train Acc', 'Train Loss','Val Acc','Val Loss'])


def train(train_loader, model, criterion, optimizer, debug=False, wmse=False):
    global device
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()
    for i, (inputs, target, meta) in enumerate(train_loader):
        input_var = inputs.to(device)
        target_var = target.to(device)

        # compute output
        output = model(input_var)
        score_map = output.cpu()

        if wmse:
            # weightes_MSE loss
            mse_weight = inputs[:,0,...].clone()
            mse_weight = mse_weight.detach()
            mse_weight = dilate_vox(mse_weight, 5)
            mse_weight = mse_weight.to(device)
            loss = criterion(output[0], target_var[0], mse_weight[0])
            for j in range(1, len(output)):
                loss += criterion(output[j], target_var[j], mse_weight[j])
        else:
            # MSE loss
            loss = criterion(output[0], target_var[0])
            for j in range(1, len(output)):
                loss += criterion(output[j], target_var[j])
        acc,_ = accuracy(score_map, target)
        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("({0:d}/{1:d}) Loss: {2:.10f} | Acc: {3: .10f}".format(i + 1, len(train_loader), losses.avg, acces.avg))
        score_map_np = score_map.clone().detach()
        if debug:  # visualize groundtruth and predictions
            sample_view = []
            for i in range(min(3, inputs.size(0))):
                inp = inputs[i].squeeze().numpy()
                tar = target[i].numpy()
                wmse = mse_weight[i].cpu().squeeze().numpy()
                scr = score_map_np[i].numpy()
                vi = three_view_with_heatmap(inp[0,...], wmse, tar, scr)
                sample_view.append(vi)
            sample_view = np.concatenate(sample_view, axis=1)
            plt.imshow(sample_view)
            #plt.show()
            plt.pause(.05)

    return losses.avg, acces.avg


def validate(val_loader, model, criterion, debug=False, wmse=False):
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (inputs, target, meta) in enumerate(val_loader):
        input_var = inputs.to(device)
        target_var = target.to(device)

        # compute output
        output = model(input_var)
        score_map = output.cpu()

        if wmse:
            #weighted_MSE loss
            mse_weight = inputs[:, 0, ...].clone()
            mse_weight = mse_weight.detach()
            mse_weight = dilate_vox(mse_weight, 3)
            mse_weight = mse_weight.to(device)
            loss = criterion(output[0], target_var[0], mse_weight[0])
            for j in range(1, len(output)):
                loss += criterion(output[j], target_var[j],mse_weight[j])
        else:
            # MSE loss
            loss = criterion(output[0], target_var[0])
            for j in range(1, len(output)):
                loss += criterion(output[j], target_var[j])
        acc,_ = accuracy(score_map, target.cpu())

        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        print("({0:d}/{1:d}) Loss: {2:.10f} | Acc: {3: .10f}".format(i + 1, len(val_loader), losses.avg, acces.avg))
        if debug:  # visualize groundtruth and predictions
            sample_view = []
            for i in range(min(3, inputs.size(0))):
                inp = inputs[i].squeeze().numpy()
                tar = target[i].numpy()
                vi = three_view_with_heatmap(inp[0,...], tar)
                sample_view.append(vi)
            sample_view = np.concatenate(sample_view, axis=1)
            plt.imshow(sample_view)
            #plt.show()
            plt.pause(.05)

    return losses.avg, acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch 3D Voxel Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vox_unet', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vox_unet)')
    parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('--workers', '-j', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=2, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=2, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')  # 2.5e-4
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('--sigma', type=float, default=4.0,
                        help='Groundtruth Gaussian sigma.')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint/checkpoint_vox_test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    #experiment params
    parser.add_argument('-wmse', dest='wmse', action='store_true', help='use weighted mes loss')
    main(parser.parse_args())
