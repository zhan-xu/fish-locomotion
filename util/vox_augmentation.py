import glob
import numpy as np
from util.mesh_obj import Mesh_obj
import os
import json

def rotation_aug(model_name, lable_name, id):
    #make sure model does not have normal
    for angle_x in range(-180, 180, 45):
        with open(lable_name, 'r') as f:
            joint_dict = json.load(f)
        print('angle_x: {}'.format(angle_x))
        model = Mesh_obj(model_name)
        arch_x = angle_x / 180 * np.pi
        rmat = np.array([[1, 0, 0, 0], [0, np.cos(arch_x), -np.sin(arch_x), 0], [0, np.sin(arch_x), np.cos(arch_x), 0], [0, 0, 0, 1]])
        for key, val in joint_dict.items():
            joint_dict[key] = np.matmul(rmat, np.append(val, 1))[:3].tolist()

        model.rotate_axis(arch_x, axis='x')
        new_name = model_name.split('/')[-1][:-4] + '_{:06d}.obj'.format(id)
        new_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl',
                                new_name)
        new_label_name = lable_name.split('/')[-1][:-4] + '_{:06d}.txt'.format(id)
        new_label_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
                                new_label_name)
        id += 1
        model.write(new_name)
        with open(new_label_name,'w') as f:
            json.dump(joint_dict,f)
    for angle_y in range(-180, 180, 45):
        if angle_y == 0:
            continue
        with open(lable_name, 'r') as f:
            joint_dict = json.load(f)
        print('angle_y: {}'.format(angle_y))
        model = Mesh_obj(model_name)
        arch_y = angle_y / 180 * np.pi
        rmat = np.array([[np.cos(arch_y), 0, np.sin(arch_y), 0], [0, 1, 0, 0], [np.sin(arch_y), 0, np.cos(arch_y), 0], [0, 0, 0, 1]])
        for key, val in joint_dict.items():
            joint_dict[key] = np.matmul(rmat, np.append(val, 1))[:3].tolist()
        model.rotate_axis(arch_y, axis='y')
        new_name = model_name.split('/')[-1][:-4] + '_{:06d}.obj'.format(id)
        new_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl',new_name)
        new_label_name = lable_name.split('/')[-1][:-4] + '_{:06d}.txt'.format(id)
        new_label_name = os.path.join(
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
            new_label_name)
        id += 1
        model.write(new_name)
        with open(new_label_name,'w') as f:
            json.dump(joint_dict,f)
    for angle_z in range(-180, 180, 45):
        if angle_z == 0:
            continue
        with open(lable_name, 'r') as f:
            joint_dict = json.load(f)
        print('angle_z: {}'.format(angle_z))
        model = Mesh_obj(model_name)
        arch_z = angle_z / 180 * np.pi
        rmat = np.array([[np.cos(arch_z), -np.sin(arch_z), 0, 0], [np.sin(arch_z), np.cos(arch_z), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for key, val in joint_dict.items():
            joint_dict[key] = np.matmul(rmat, np.append(val, 1))[:3].tolist()
        model.rotate_axis(arch_z, axis='z')
        new_name = model_name.split('/')[-1][:-4] + '_{:06d}.obj'.format(id)
        new_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl',new_name)
        new_label_name = lable_name.split('/')[-1][:-4] + '_{:06d}.txt'.format(id)
        new_label_name = os.path.join(
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
            new_label_name)
        id += 1
        model.write(new_name)
        with open(new_label_name,'w') as f:
            json.dump(joint_dict,f)

    return id


def scale_aug(model_name, lable_name, id):
    for scale_x in [0.7, 0.85, 1, 1.2, 1.4]:
        print('scale_x: {}'.format(scale_x))
        with open(lable_name, 'r') as f:
            joint_dict = json.load(f)
        model = Mesh_obj(model_name)
        model.scale_axis(scale_x, axis='x')
        for key, val in joint_dict.items():
            joint_dict[key][0] = joint_dict[key][0]*scale_x
        new_name = model_name.split('/')[-1][:-4] + '_{:06d}.obj'.format(id)
        new_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl',
                                new_name)
        new_label_name = lable_name.split('/')[-1][:-4] + '_{:06d}.txt'.format(id)
        new_label_name = os.path.join(
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
            new_label_name)
        with open(new_label_name,'w') as f:
            json.dump(joint_dict,f)
        id += 1
        model.write(new_name)

    for scale_y in [0.7, 0.85, 1.2, 1.4]:
        print('scale_y: {}'.format(scale_y))
        with open(lable_name, 'r') as f:
            joint_dict = json.load(f)
        model = Mesh_obj(model_name)
        model.scale_axis(scale_y, axis='y')
        for key, val in joint_dict.items():
            joint_dict[key][1] = joint_dict[key][1]*scale_y
        new_name = model_name.split('/')[-1][:-4] + '_{:06d}.obj'.format(id)
        new_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl',
                                new_name)
        new_label_name = lable_name.split('/')[-1][:-4] + '_{:06d}.txt'.format(id)
        new_label_name = os.path.join(
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
            new_label_name)
        with open(new_label_name, 'w') as f:
            json.dump(joint_dict, f)
        id += 1
        model.write(new_name)

    for scale_z in [0.7, 0.85, 1.2, 1.4]:
        print('scale_z: {}'.format(scale_z))
        with open(lable_name, 'r') as f:
            joint_dict = json.load(f)
        model = Mesh_obj(model_name)
        model.scale_axis(scale_z, axis='z')
        for key, val in joint_dict.items():
            joint_dict[key][2] = joint_dict[key][2]*scale_z
        new_name = model_name.split('/')[-1][:-4] + '_{:06d}.obj'.format(id)
        new_name = os.path.join('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl',
                                new_name)
        new_label_name = lable_name.split('/')[-1][:-4] + '_{:06d}.txt'.format(id)
        new_label_name = os.path.join(
            '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval_rot_scl',
            new_label_name)
        with open(new_label_name, 'w') as f:
            json.dump(joint_dict, f)
        id += 1
        model.write(new_name)
    return id

if __name__ == '__main__':
    model_names = glob.glob('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval/tgrshark_*.obj')
    label_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/labels/trainval/'
    for model_name in model_names:
        print(model_name)
        lable_name = model_name.split('/')[-1].replace('.obj','.txt')
        lable_name = os.path.join(label_folder, lable_name)
        id = 1
        id = rotation_aug(model_name, lable_name, id)
        id = scale_aug(model_name, lable_name, id)