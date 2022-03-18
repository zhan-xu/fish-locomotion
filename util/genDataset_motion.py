import json
import cv2
import numpy as np
import glob
import os
import random
import shutil

joint_order = ['sp0', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6', 'hd1', 'hd2','hd3']


def genJson(img_folder, txt_folder, json_name):
    subset_name = img_folder.split('/')[-3]
    # Obtain an empty dict for useless field
    jsonfile = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/mpii/mpii_annotations.json'
    with open(jsonfile) as anno_file:
        anno = json.load(anno_file)
        a = anno[11]
        empty_dict = a['objpos_other']

    new_anno = []
    images = glob.glob(img_folder + '*.jpg')
    print ("processing: " + img_folder)
    print ("Total image number: ", len(images))
    num_total_image = len(images)
    random.shuffle(images)

    # for i in range(1,9623):
    #     img_src = images[i]
    #     img_dst = img_src.replace('/trainval','/test')
    #     shutil.move(img_src,img_dst)
    #     txt_file_src = img_src.split('/')[-1].replace('.jpg', '.txt')
    #     txt_file_src = os.path.join('/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/labels', txt_file_src)
    #     txt_file_dst = txt_file_src.replace('/trainval','/test')
    #     #print 'a'
    #     shutil.move(txt_file_src,txt_file_dst)

    ind_img = 1
    mean_val = np.zeros(3)
    std_val = np.zeros(3)
    for img_file in images:
        #print ind_img, img_file
        #img_file = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/images/0000051636.jpg'
        img_anno = {}
        img_anno['img_paths'] = subset_name + '/images_binary/'+img_file.split('/')[-1]
        txt_file = img_file.split('/')[-1].replace('.jpg', '.txt')
        txt_file = os.path.join(txt_folder, txt_file)
        # print img_file
        # print txt_file
        img = cv2.imread(img_file)
        (h, w, _) = img.shape
        img_anno['img_width'] = w
        img_anno['img_height'] = h
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = np.nonzero(gray_image)
        x_min = max(np.min(x) - 20, 0)
        x_max = min(np.max(x) + 20, h)
        y_min = max(np.min(y) - 20, 0)
        y_max = min(np.max(y) + 20, w)

        # cv2.rectangle(img,(y_min, x_min),(y_max, x_max),(0,255,0),3)
        # cv2.namedWindow("kkk")
        # cv2.imshow("kkk",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mean_val += img[x_min:x_max, y_min:y_max, :].reshape(-1,3).mean(axis = 0)
        std_val += img[x_min:x_max, y_min:y_max, :].reshape(-1,3).std(axis = 0)

        max_dim = np.max(np.array([x_max - x_min, y_max - y_min]))
        # print max_dim
        s = max_dim / 200.0
        c = [(y_max + y_min) / 2.0, (x_max + x_min) / 2.0]
        img_anno['scale_provided'] = s
        img_anno['objpos'] = c

        label_file = open(txt_file, 'r')
        pos = json.load(label_file)
        # print pos
        label_file.close()
        pts = []
        for j in range(len(joint_order)):
            key = joint_order[j]
            cv2.circle(img, (int(pos[key][0]), int(pos[key][1])), 4, [255, 255, 255], 2)
            if pos[key][0] < 0 or pos[key][0] >= w or pos[key][1] < 0 or pos[key][1] >= h:
                pts.append([0.0, 0.0, 0.0])
            else:
                pts.append([pos[key][0], pos[key][1], 1.0])

        img_anno['joint_self'] = pts
        img_anno['joint_others'] = empty_dict
        img_anno['objpos_other'] = empty_dict
        img_anno['scale_provided_other'] = empty_dict
        img_anno['numOtherPeople'] = 0.0
        # split into train and val
        if ind_img < 0.15*num_total_image:
           img_anno['isValidation'] = 1.0
        else:
           img_anno['isValidation'] = 0.0
        #img_anno['isValidation'] = 1.0
        img_anno['people_index'] = 1.0
        img_anno['dataset'] = 'Shark'
        n_img = int(img_anno['img_paths'][-14:-4])
        img_anno['annolist_index'] = n_img
        new_anno.append(img_anno)
        ind_img += 1

    with open(json_name, 'w') as outfile:
        json.dump(new_anno, outfile)

    print (len(images))
    print (mean_val / len(images))
    print (std_val / len(images))


def fabricateImg(ori_folder, dst_folder):
    images = glob.glob(ori_folder + '*.jpg')
    for img_file in images:
        print(img_file)
        img = cv2.imread(img_file, 0)
        # gray scale
        #img_res = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #img_res[np.where(img_res<=10)] = 100

        # binary
        _, img_res = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

        # show image
        # cv2.namedWindow('1')
        # cv2.imshow('1', img)
        # cv2.namedWindow('2')
        # cv2.imshow('2', img_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img_file_new = img_file.replace(ori_folder,dst_folder)
        #print img_file_new
        cv2.imwrite(img_file_new,img_res)


def scale_augment(img_list, tar_folder, start_idx):
    joint_list = ['hd1', 'hd2', 'hd3', 'sp0', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6']
    for img_filename in img_list:
        print(img_filename)
        txt_filename = img_filename.replace('.jpg', '.txt')
        txt_filename = txt_filename.replace('images', 'labels')
        print (txt_filename)
        img = cv2.imread(img_filename)
        with open(txt_filename, 'r') as anno_file:
            anno = json.load(anno_file)


        for sc in [(1.6, 1.0), (1, 1), (1, 1.6), (0.7, 1), (1, 0.7)]:
            sc_real = sc+np.random.uniform(-0.2, 0.2, 2)
            img_dst = cv2.resize(img, (0, 0), fx=sc_real[0], fy=sc_real[1], interpolation=cv2.INTER_CUBIC)
            json_dst = {}
            for jt in joint_list:
                anno_real = (anno[jt][0] * sc_real[0], anno[jt][1] * sc_real[1])
                json_dst[jt] = [anno_real[0],anno_real[1]]

            # for jt in joint_list:
            #     cv2.circle(img_dst, (int(json_dst[jt][0]), int(json_dst[jt][1])), 4, (0, 0, 255), 2)
            # cv2.namedWindow('img_ori')
            # cv2.imshow('img_ori', img)
            # cv2.namedWindow('img_dst')
            # cv2.imshow('img_dst', img_dst)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            tar_img_filename = tar_folder + 'images/{:010d}.jpg'.format(start_idx)
            tar_txt_filename = tar_folder + 'labels/{:010d}.txt'.format(start_idx)
            cv2.imwrite(tar_img_filename, img_dst)
            with open(tar_txt_filename, 'w') as anno_outfile:
                json.dump(json_dst,anno_outfile)
            start_idx += 1
    return start_idx


def pick_dataset():
    start_idx = 1
    print (start_idx)
    images = glob.glob('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/images/*.jpg')
    random.shuffle(images)
    images = images[:12000]
    start_idx = scale_augment(images, tar_folder='/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval_new/',
                              start_idx=start_idx)
    print(start_idx)

    images = glob.glob('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval2/images/*.jpg')
    start_idx = scale_augment(images, tar_folder='/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval2_new/',
                              start_idx=start_idx)
    print(start_idx)

    images = glob.glob('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data/images/*.jpg')
    start_idx = scale_augment(images,
                              tar_folder='/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data_new/',
                              start_idx=start_idx)
    print(start_idx)

    images = glob.glob('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/SandTiger_training_data/images/*.jpg')
    start_idx = scale_augment(images,
                              tar_folder='/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/SandTiger_training_data_new/',
                              start_idx=start_idx)
    print(start_idx)

    images = glob.glob('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/tgrshark_training_data/images/*.jpg')
    start_idx = scale_augment(images,
                              tar_folder='/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/tgrshark_training_data_new/',
                              start_idx=start_idx)
    print(start_idx)

if __name__ == '__main__':
    # with open('shark_annotations.json') as anno_file:
    #     anno = json.load(anno_file)

    #fabricateImg('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/tgrshark_training_data_new/images/',
    #            '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/tgrshark_training_data_new/images_binary/')
    fabricateImg('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/example/fish_snapshot/',
                 '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/example/fish_snapshot_bin/')
    #fabricateImg('/home/zhanxu/Proj/pytorch-pose/data/shark/test/images/',
    #             '/home/zhanxu/Proj/pytorch-pose/data/shark/test/images_binary/')

    #genJson('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval2/images_binary/',
    #       '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval2/labels',
    #       'shark_annotations_trainval2.json')
    #genJson('/home/zhanxu/Proj/pytorch-pose/data/shark/test/images/',
    #        '/home/zhanxu/Proj/pytorch-pose/data/shark/test/labels',
    #        'shark_annotations_test.json')


    # for i in range(1,3601):
    #     src_filename = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data/images/{:010d}.jpg'.format(i)
    #     dst_filename = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data/images/{:010d}.jpg'.format(i + 76368)
    #     shutil.move(src_filename, dst_filename)
    #     src_filename = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data/labels/{:010d}.txt'.format(i)
    #     dst_filename = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data/labels/{:010d}.txt'.format(i + 76368)
    #     shutil.move(src_filename,dst_filename)


    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/shark_annotations_trainval.json','r') as anno:
    #     list1 = json.load(anno)
    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval2/shark_annotations_trainval2.json','r') as anno2:
    #     list2 = json.load(anno2)
    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/chark_training_data/shark_annotations_chark.json','r') as anno3:
    #     list3 = json.load(anno3)
    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/SandTiger_training_data/shark_annotations_SandTiger.json','r') as anno4:
    #     list4 = json.load(anno4)
    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/tgrshark_training_data/shark_annotations_tgrshark.json','r') as anno5:
    #     list5 = json.load(anno5)
    # print len(list1)
    # print len(list2)
    # print len(list3)
    # print len(list4)
    # print len(list5)
    # list_all = list1 + list2 + list3 + list4 + list5
    # print len(list_all)
    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/shark_annotations_trainval_all.json', 'w') as outfile:
    #     json.dump(list_all, outfile)

    #with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/shark_annotations_trainval2.json', 'r') as anno:
    #    list1 = json.load(anno)
    # for item in list1:
    #     item['img_paths'] = item['img_paths'][1:]
    # with open('/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/shark_annotations_trainval_new.json', 'w') as outfile:
    #     json.dump(list1, outfile)

