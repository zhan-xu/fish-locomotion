#--resume ../checkpoint/shark/h2_augdata/model_best.pth.tar --color_mode L --meanstd ../data/shark/mean_bin.pth.tar --imgpath ./fish_snapshot_bin/
#--resume ../checkpoint/shark/angle_new/model_best.pth.tar --color_mode L --meanstd ../data/shark/mean_bin.pth.tar --imgpath ./synthetic_frame_bin/ --reg='angle'
from __future__ import print_function, absolute_import

import argparse
import glob

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from util.osutils import isfile
from models.motionNet.regnet import regnet

from util.transforms import *
from util.evaluation_util import get_preds_motion

import cv2
import numpy as np
import os

def main(args):
    model = regnet(color_mode=args.color_mode, num_classes=args.num_classes)

    model = torch.nn.DataParallel(model).cuda()
    #model = model.cuda()

    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    print('\nEvaluation only')
    #predictions = test(args.imgpath, model, args.num_classes,  args.meanstd, args.color_mode, args.reg)
    predictions = test_different_image(args.imgpath, model, args.num_classes, args.meanstd, args.color_mode, args.reg)
    np.save('preds_model_9.npy',predictions)
    np.savetxt('preds_model_9.txt', predictions,fmt='%.04f')
    #predictions = np.load('preds.npy')
    #print(predictions.shape)
    #predictions_sm = temporal_smooth(predictions)
    #np.save('preds_sm.npy', predictions_sm)
    return


def getMeanStd(imgList):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img_path in imgList:
        img = load_image(img_path)  # CxHxW
        mean += img.view(img.size(0), -1).mean(1)
        std += img.view(img.size(0), -1).std(1)
    mean /= len(imgList)
    std /= len(imgList)
    return mean, std


def test_different_image(img_folder, model, num_cls, meanstd_file, color_mode, reg):
    model.eval()
    if isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)
        mean = meanstd['mean']
        std = meanstd['std']
    else:
        print('no file found for mean and variance.')
        exit(0)
    inp_res = 256
    img_list = glob.glob(os.path.join(img_folder+'*.jpg'))

    num_frame = 1
    total_frame = len(img_list)
    if reg == 'heatmap':
        total_pred = np.zeros((total_frame, num_cls, 2))
    else:
        total_pred = np.zeros((total_frame, num_cls-1))
    for img_id in range(total_frame):
    #for img_path in img_list:
        img_path = img_folder+'frame_{:d}.jpg'.format(img_id)
        #img_path = img_folder+'mask_{:03d}.jpg'.format(i)
        print(img_path)
        img = load_image(img_path)  # CxHxW
        inp, c, s = simple_crop(img, inp_res)

        '''img_np = inp.cpu().numpy()
        img_np = np.transpose(img_np, (1,2,0))
        cv2.namedWindow('kk')
        cv2.imshow('kk', img_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        inp = color_normalize(inp, mean, std)
        if color_mode!='RGB':
            inp = torch.mean(inp, dim=0)
            inp = inp>0.05
            inp = inp.float()
            inp = inp[None,:,:]
        inp = inp[None, ...]

        '''sim = np.transpose(inp.cpu().numpy().squeeze(axis=0), (1, 2, 0))
        cv2.namedWindow('kk')
        cv2.imshow('kk', sim)
        cv2.waitKey()
        cv2.destroyAllWindows()'''

        import time
        start = time.time()
        input_var = torch.autograd.Variable(inp.cuda(), volatile=True)
        output = model(input_var)

        if reg == 'heatmap':
            score_map = output[-1].data.cpu()
            preds = simple_final_preds(score_map, c, s, [64, 64])
            preds_np = preds.numpy()
            img_draw = cv2.imread(img_path)
            for p in range(preds_np.shape[0]):
                cv2.circle(img_draw, (int(preds_np[p, 0]), int(preds_np[p, 1])), 3, (0, 0, 255), 3)
            # cv2.namedWindow('joints')
            # cv2.imshow('joints',img_draw)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            filename = img_path.split('/')[-1]
            #filename = 'res_{:03d}.png'.format(num_frame)
            cv2.imwrite(filename, img_draw)
            total_pred[num_frame - 1, ...] = preds_np
        else:
            score_map = output.data.cpu()
            total_pred[num_frame-1,:] = score_map.numpy().squeeze()
        num_frame += 1
        end = time.time()
        print(end-start)
        break
    return total_pred

def test(img_folder, model, num_cls, meanstd_file, color_mode, reg):
    model.eval()
    if isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)
        mean = meanstd['mean']
        std = meanstd['std']
    else:
        print('no file found for mean and variance.')
        exit(0)
    inp_res = 256
    img_list = glob.glob(os.path.join(img_folder+'*.jpg'))

    num_frame = 1
    total_frame = len(img_list)
    if reg == 'heatmap':
        total_pred = np.zeros((total_frame, num_cls, 2))
    else:
        total_pred = np.zeros((total_frame, num_cls-1))
    for i in range(1,total_frame+1):
        #img_path = img_folder+'mask_{:03d}.jpg'.format(i)
        img_path = img_folder + '{:04d}.jpg'.format(i)
        print(img_path)
        img = load_image(img_path)  # CxHxW
        inp, c, s = simple_crop(img, inp_res)

        '''img_np = inp.cpu().numpy()
        img_np = np.transpose(img_np, (1,2,0))
        cv2.namedWindow('kk')
        cv2.imshow('kk', img_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        inp = color_normalize(inp, mean, std)
        if color_mode!='RGB':
            inp = torch.mean(inp, dim=0)
            inp = inp>0.05
            inp = inp.float()
            inp = inp[None,:,:]
        inp = inp[None, ...]

        '''sim = np.transpose(inp.cpu().numpy().squeeze(axis=0), (1, 2, 0))
        cv2.namedWindow('kk')
        cv2.imshow('kk', sim)
        cv2.waitKey()
        cv2.destroyAllWindows()'''

        input_var = torch.autograd.Variable(inp.cuda(), volatile=True)
        output = model(input_var)

        if reg == 'heatmap':
            score_map = output[-1].data.cpu()
            preds = simple_final_preds(score_map, c, s, [64, 64])
            preds_np = preds.numpy()
            img_draw = cv2.imread(img_path)
            for p in range(preds_np.shape[0]):
                cv2.circle(img_draw, (int(preds_np[p, 0]+40), preds_np[p, 1]), 3, (0, 0, 255), 5)
            # cv2.namedWindow('joints')
            # cv2.imshow('joints',img_draw)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            filename = 'res_{:03d}.png'.format(num_frame)
            cv2.imwrite(filename, img_draw)
            total_pred[num_frame - 1, ...] = preds_np+[40,0]
        else:
            score_map = output.data.cpu()
            total_pred[i-1,:] = score_map.numpy().squeeze()
        num_frame += 1
    return total_pred


def simple_crop(img, res):
    img_numpy = im_to_numpy(img)
    (h,w,_) = img_numpy.shape
    gray_image = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2GRAY)
    # im_bw = cv2.threshold(gray_image[40:,...], 0.025, 255, cv2.THRESH_BINARY)[1]
    # cv2.namedWindow('1')
    # cv2.imshow('1',im_bw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #x, y = np.nonzero(gray_image[40:,...]>0.05)
    x, y = np.nonzero(gray_image> 0.05)
    x_min = max(np.min(x)-5, 0)
    x_max = min(np.max(x)+5, h)
    y_min = max(np.min(y)-5, 0)
    y_max = min(np.max(y)+5, w)

    #for real video
    # x_min = 280
    # x_max = 570
    # y_min = 0
    # y_max = 950

    max_dim = np.max(np.array([x_max - x_min, y_max - y_min]))
    # print max_dim
    s = max_dim / 200.0
    c = [(y_max + y_min) / 2.0, (x_max + x_min) / 2.0]
    c = torch.Tensor(c)
    inp = crop(img, c, s, [res, res])
    return inp, c, s


def simple_final_preds(output, center, scale, res):
    coords = get_preds_motion(output) # float type
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
    # Transform back
    preds = transform_preds(coords[0], center, scale, res)

    return preds


def temporal_smooth(pred_ori):
    t,p,_ = pred_ori.shape
    pred_res = pred_ori.copy()
    for tt in range(4,t-4):
        #pred_res[tt,:,:] = np.median(pred_ori[tt-3:tt+4,:,:], axis=0)
        pred_res[tt, :, :] = np.mean(pred_ori[tt - 4:tt + 5, :, :], axis=0)
    #display results
    for f in range(140):
        img_name = 'synthetic_frame_bin/{:04d}.jpg'.format(f+1)
        img = cv2.imread(img_name)
        img2 = img.copy()
        for p in range(pred_ori.shape[1]):
            cv2.circle(img,(int(pred_ori[f,p,0]), int(pred_ori[f,p,1])), 3, (0,0,255), 5)
            cv2.circle(img2, (int(pred_res[f, p, 0]), int(pred_res[f, p, 1])), 3, (0, 0, 255), 5)
        img_con = np.concatenate((img,img2), axis=0)
        filename = 'sythetic_res_{:03d}.png'.format(f+1)
        cv2.imwrite(filename, img_con)
    return pred_res


if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=10, type=int, metavar='N',
                        help='Number of keypoints')
    parser.add_argument('--resume', default='./checkpoint/angle_classic/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--imgpath', default='./test_motion_seq/model_9/', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--color_mode', default='L', type=str, help='color mode of feed-in images')
    parser.add_argument('--meanstd',
                        default='/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/motion_pred/mean_bin.pth.tar',
                        type=str, help='mean std file')
    parser.add_argument('--reg', default='angle', type=str, help='regression target type')

    parser.parse_args()
    main(parser.parse_args())
