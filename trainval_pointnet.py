import numpy as np
import argparse
import torch

from datasets import Fish_Ply
from models.pointNet.pointnet import PointNetReg as Pointnet

from util.osutils import mkdir_p, isfile, join
from util.logger import Logger
from util.training_util import adjust_learning_rate, save_checkpoint, three_view_with_pointnet
from util.evaluation_util import accuracy_coord, AverageMeter

best_acc = 0
device = None


def main(args):
    global best_acc
    global device

    # create checkpoint dir
    mkdir_p(args.checkpoint)

    # create model
    print("==> creating pointnet as backbone")
    model = Pointnet(k=3*args.num_classes, num_points = 1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs.')
        model = torch.nn.DataParallel(model)
    model.to(device)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.MSELoss(reduction='elementwise_mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'rigging network'
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
        Fish_Ply(args.json_file, args.img_folder, sample_num=1000), batch_size=args.train_batch,
        shuffle=True, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        Fish_Ply(args.json_file, args.img_folder, sample_num=1000, train=False), batch_size=args.test_batch,
        shuffle=False, num_workers=args.workers, pin_memory=True)

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
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args.debug)

        # evaluate on validation set
        valid_loss, valid_acc = validate(val_loader, model, criterion, args.debug)

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


def train(train_loader, model, criterion, optimizer, debug=False):
    global device
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()
    for i, (inputs, target, meta) in enumerate(train_loader):
        if debug:  # visualize groundtruth and predictions
            for i in range(inputs.size(0)):
                inp = inputs[i].squeeze().numpy()
                tar = target[i].numpy()
                three_view_with_pointnet(inp, tar)

        if inputs.size(0) == 1:
            inputs = np.repeat(inputs, 2, axis=0)
            target = np.repeat(target, 2, axis=0)

        inputs = np.transpose(inputs,(0,2,1))
        input_var = inputs.to(device)
        target_var = target.view(target.shape[0], -1).to(device)

        # compute output
        output,_ = model(input_var)
        score_map = output.data.cpu()
        loss = criterion(output, target_var)
        acc = accuracy_coord(score_map, target_var.data.cpu())
        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("({0:d}/{1:d}) Loss: {2:.10f} | Acc: {3: .10f}".format(i + 1, len(train_loader), losses.avg, acces.avg))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion, debug=False):
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (inputs, target, meta) in enumerate(val_loader):
        if debug:  # visualize groundtruth and predictions
            for i in range(inputs.size(0)):
                inp = inputs[i].squeeze().numpy()
                tar = target[i].numpy()
                three_view_with_pointnet(inp, tar)

        if inputs.size(0) == 1:
            inputs = np.repeat(inputs, 2, axis=0)
            target = np.repeat(target, 2, axis=0)
        inputs = np.transpose(inputs, (0, 2, 1))
        input_var = inputs.to(device)
        target_var = target.view(target.shape[0], -1).to(device)

        # compute output
        output,_ = model(input_var)
        score_map = output.data.cpu()
        loss = criterion(output, target_var)
        acc = accuracy_coord(score_map, target_var.data.cpu())

        # measure accuracy and record loss
        losses.update(loss.data, inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        print("({0:d}/{1:d}) Loss: {2:.10f} | Acc: {3: .10f}".format(i + 1, len(val_loader), losses.avg, acces.avg))

    return losses.avg, acces.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet based Rigging Network')
    # Model structure
    parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('--workers', '-j', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=16, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')  # 2.5e-4
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Miscs
    # parser.add_argument('-c', '--checkpoint', default='checkpoint/pointnet_test', type=str, metavar='PATH',
    #                     help='path to save checkpoint (default: checkpoint)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint/pointnet_test', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')


    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    # file path
    parser.add_argument('--json_file',
                        default='/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_ply_annotations_trainval.json',
                        type=str, help='json file path')
    parser.add_argument('--img_folder',
                        default='/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data',
                        type=str, help='img folder')
    main(parser.parse_args())
