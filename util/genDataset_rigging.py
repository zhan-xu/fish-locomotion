import json
import numpy as np
import glob
import os
import random

import util.binvox_rw as binvox_rw
from util.vox_util import Cartesian2Voxcoord
from util.mesh_obj import Mesh_obj

joint_order = ['hd1', 'hd2','hd3', 'sp0', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6']


def gen_json_vox(img_folder, txt_folder, json_name, split=True):
    new_anno = []
    images = glob.glob(img_folder + '*.binvox')
    print("processing: " + img_folder)
    print("Total image number: ", len(images))
    num_total_image = len(images)
    random.shuffle(images)
    ind_img = 1
    for img_file in images:
        print(ind_img)
        #print ind_img, img_file
        #img_file = '/mnt/gypsum/home/zhanxu/Proj/pytorch-pose/data/shark/trainval/images/0000051636.jpg'
        img_anno = {}
        if split:
            img_anno['img_paths'] = 'voxel/trainval_64/'+img_file.split('/')[-1]
        else:
            img_anno['img_paths'] = 'voxel/test/aug/' + img_file.split('/')[-1]
        txt_file = img_file.split('/')[-1].replace('.binvox', '.txt')
        txt_file = os.path.join(txt_folder, txt_file)
        # print img_file
        # print txt_file
        with open(img_file, 'rb') as f:
            mesh_vox = binvox_rw.read_as_3d_array(f)
        if np.max(mesh_vox.data.astype(np.float32)) <= 0.001 or np.sum(mesh_vox.data.astype(np.float32)) < 1000:
            continue
        img_anno['dims'] = mesh_vox.dims
        img_anno['translate'] = mesh_vox.translate
        img_anno['scale'] = mesh_vox.scale

        label_file = open(txt_file, 'r')
        pos = json.load(label_file)
        # print pos
        label_file.close()
        pts = []
        for j in range(len(joint_order)):
            key = joint_order[j]
            vox_cood = Cartesian2Voxcoord(np.asarray(pos[key]), mesh_vox.translate, mesh_vox.scale, resolution=(64,64,64))
            vox_cood = np.clip(vox_cood,0, mesh_vox.dims[0]-1)
            pts.append(vox_cood.tolist())
        img_anno['joint'] = pts
        if split:
            # split into train and val
            if ind_img < 0.15*num_total_image:
               img_anno['isValidation'] = 1.0
            else:
               img_anno['isValidation'] = 0.0
        else:
            img_anno['isValidation'] = 1.0
        new_anno.append(img_anno)
        ind_img += 1

    with open(json_name, 'w') as outfile:
        json.dump(new_anno, outfile)


def gen_json_ply(model_folder, txt_folder, json_name, split=True):
    new_anno = []
    models = glob.glob(model_folder + '*.obj')
    print("processing: " + model_folder)
    print("Total model number: ", len(models))
    num_total_image = len(models)
    random.shuffle(models)
    ind_img = 1
    for model_file in models:
        print(ind_img)
        img_anno = {}
        if split:
            img_anno['img_paths'] = 'ply/trainval/' + model_file.split('/')[-1]
        else:
            img_anno['img_paths'] = 'ply/test/' + model_file.split('/')[-1]
        txt_file = model_file.split('/')[-1].replace('.obj', '.txt')
        txt_file = os.path.join(txt_folder, txt_file)
        model = Mesh_obj(model_file)
        if np.max(model.v.shape[0]) < 800:
            continue
        label_file = open(txt_file, 'r')
        pos = json.load(label_file)
        # print pos
        label_file.close()
        pts = []
        for j in range(len(joint_order)):
            key = joint_order[j]
            pts.append(pos[key])
        img_anno['joint'] = pts
        if split:
            # split into train and val
            if ind_img < 0.1*num_total_image:
              img_anno['isValidation'] = 1.0
            else:
              img_anno['isValidation'] = 0.0
        else:
            img_anno['isValidation'] = 1.0
        new_anno.append(img_anno)
        ind_img += 1

    with open(json_name, 'w') as outfile:
        json.dump(new_anno, outfile)


if __name__ == '__main__':
    '''gen_json_vox('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64/',
           '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
           '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_vox_annotations_trainval_64.json')'''
    '''gen_json_vox('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test/aug/',
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/test/aug',
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_vox_annotations_test2.json')'''

    gen_json_ply('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/ply/trainval/',
                 '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval',
                 '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_ply_annotations_trainval.json')

    '''gen_json_ply('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/ply/test/',
                 '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/test',
                 '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/fish_ply_annotations_test.json',
                 split=False)'''
