import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2


def filter_vector(pts): #unfinished...
    pts_new = pts.copy()
    #pts_new = np.zeros(pts.shape)
    n,d = pts.shape
    A = np.eye(d)
    Q = 0.004 * np.eye(d)
    R = 0.01 * np.eye(d)
    p = 1.0 * np.eye(d)
    for i in range(1, n):
        z = pts[i,:].reshape(-1, 1)
        x_pre = pts_new[i-1,:].reshape(-1, 1)
        dir = (z - x_pre) / np.linalg.norm(z - x_pre)
        x_ = np.matmul(A,x_pre) + 0.3*dir
        p_ = np.matmul(np.matmul(A,p), A.T) + Q
        k = np.diag(np.divide(np.diag(p_), (np.diag(p_) + np.diag(R))))
        x = x_ + np.matmul(k, (z-x_))
        p = (np.eye(2) - k)*p_
        pts_new[i,:]=x.squeeze()
        #print p_[0,0], k[0,0], x[0], p[0,0]
    return pts_new


def filter_scalar(x):
    x_res = np.zeros(x.shape)
    x_res[0] = x[0]
    x_pre = x[0]
    p_pre = 1.0
    #r = 0.01
    #q = 0.004
    r = 0.1
    q = 0.01
    for k in range(1,len(x)):
        z_k = x[k]
        dir = (z_k - x_pre) / np.linalg.norm(z_k - x_pre)
        #x_k_prim = x_pre + 0.3*dir
        x_k_prim = x_pre
        p_k_prim = p_pre + q
        k_k = p_k_prim / (p_k_prim + r)

        x_pre = x_k_prim + k_k * (z_k - x_k_prim)
        x_res[k] = x_pre
        p_pre = (1-k_k)*p_k_prim
        print p_k_prim, k_k, x_pre, p_pre

    x_2 = x.copy()
    for i in range(3, len(x)-3):
        x_2[i] = np.mean(x[i-3:i+3])
    line1, = plt.plot(x,'r',label="original x")
    line2, = plt.plot(x_res,'g',label="karman filter")
    line3, = plt.plot(x_2,'b',label="temporal average",)
    plt.legend(loc='upper right')
    plt.show()
    return x_res


if __name__ == "__main__":
    data = np.load('preds.npy')
    data_est = np.zeros(data.shape)
    '''for p in range(data.shape[1]):
        x = data[:,p,:].copy()
        data_est[:,p,:] = filter_vector(x)
    np.save('preds_km.npy', data_est)'''
    #plt.plot(x[:, 1], 'r', x_est[:, 1], 'g')
    #plt.show()

    x = data[:, 6, :].copy()
    x_est = filter_scalar(x[:,0])

    #show results
    '''img_folder = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose-pycharm/example/synthetic_frame'
    img_list = glob.glob(os.path.join(img_folder,'*.jpg'))
    num_frame = 1
    pred_km = np.load('preds_km.npy')
    pred_ts = np.load('preds_sm.npy')
    pred_ori = np.load('preds.npy')
    # mean, std = getMeanStd(img_list)
    f = 0
    for img_path in img_list:
        print(img_path)
        img = cv2.imread(img_path)
        img2 = img.copy()
        img3 = img.copy()
        for p in range(pred_ori.shape[1]):
            cv2.circle(img, (int(pred_ori[f, p, 0]), int(pred_ori[f, p, 1])), 3, (0, 0, 255), 5)
            cv2.circle(img2, (int(pred_ts[f, p, 0]), int(pred_ts[f, p, 1])), 3, (0, 0, 255), 5)
            cv2.circle(img3, (int(pred_km[f, p, 0]), int(pred_km[f, p, 1])), 3, (0, 0, 255), 5)
        img_con = np.concatenate((img, img2, img3), axis=0)
        # cv2.namedWindow('joints')
        # cv2.imshow('joints',img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        filename = 'res_{:03d}.png'.format(f+1)
        cv2.imwrite(filename, img_con)
        f += 1'''