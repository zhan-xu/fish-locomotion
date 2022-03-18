'''
python=3.6.6
'''
import glob
import numpy as np
import os
from util.mesh_obj import Mesh_obj
import util.binvox_rw as binvox_rw
from util.vox_util import Cartesian2Voxcoord
import shutil


def binvox():
    #src_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl/'
    src_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/test/aug/'
    src_models = glob.glob(src_folder + '*.obj')
    #dst_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_128/'
    dst_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test/aug/'
    for src_model_name in src_models:
        dst_model = src_model_name.replace(src_folder, dst_folder).replace('.obj', '.binvox')
        if os.path.exists(dst_model):
            #with open(src_model_name.replace('.obj', '.binvox'), 'rb') as f:
            #    try:
            #        model = binvox_rw.read_as_3d_array(f)
            #        print(model.data.shape)
            #    except:
            #        os.system("./binvox -d 256 -pb" + src_model_name)
            #    else:
            #        continue
            continue
        else:
            print(src_model_name)
            os.system("./binvox -d 64 -pb " + src_model_name)
            shutil.move(src_model_name.replace('.obj', '.binvox'), dst_model)


def convert_vox_to_points():
    with open('blue_shark_030_000015.binvox', 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    f_out = open('blue_shark_030_000015_conv.obj', 'w')
    scale = model.scale
    translate = model.translate
    for i in range(model.data.shape[0]):
        for j in range(model.data.shape[1]):
            for k in range(model.data.shape[2]):
                if model.data[i, j, k]:
                    x = (i + .5) / model.data.shape[0] * scale + translate[0]
                    y = (j + .5) / model.data.shape[1] * scale + translate[1]
                    z = (k + .5) / model.data.shape[2] * scale + translate[2]
                    f_out.write('v ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
    f_out.close()


def surface_sample_supply():
    obj_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/test/aug/'
    sample_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/surface_sample_tmp/'
    vox_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test/aug/'
    obj_list = glob.glob(obj_folder + '*.obj')
    obj_idx=0
    for obj_name in obj_list:
        obj_idx+=1
        print('{:}/{:}'.format(obj_idx, len(obj_list)))
        obj_name_new = obj_name.replace(obj_folder,sample_folder)
        vox_name = obj_name.replace(obj_folder, vox_folder)[:-3] + 'binvox'
        #vox_folder_new = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64_resample/'
        vox_folder_new = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test_resample/'
        vox_name_new = vox_name.replace(vox_folder, vox_folder_new)
        if os.path.exists(vox_name_new):
            continue
        if not os.path.exists(obj_name_new):
            os.system('meshlabserver -i '+ obj_name + ' -o ' + obj_name_new + ' -s stratified_triangle_sampling.mlx')
        mesh_sursamp = Mesh_obj(obj_name_new)
        try:
            with open(vox_name, 'rb') as f:
                mesh_vox = binvox_rw.read_as_3d_array(f)
        except:
            continue
        trans = mesh_vox.translate
        scl = mesh_vox.scale
        debug_vox = np.zeros(shape=(64, 64, 64), dtype=np.bool) #debug
        for i in range(mesh_sursamp.v.shape[0]):
            v = mesh_sursamp.v[i,:]
            (vx,vy,vz) = Cartesian2Voxcoord(v, translate = trans, scale = scl, resolution = np.asarray(mesh_vox.dims))
            try:
                debug_vox[vx,vy,vz] = True
                if mesh_vox.data[vx,vy,vz]:
                    continue
                else:
                    mesh_vox.data[vx,vy,vz] = True
            except:
                continue
        vox_new = binvox_rw.Voxels(mesh_vox.data, mesh_vox.dims, mesh_vox.translate, mesh_vox.scale, mesh_vox.axis_order)

        with open(vox_name_new, 'wb') as fw:
            vox_new.write(fw)
    return


def thinvox():
    #src_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64/'
    src_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test/aug/'
    src_models = glob.glob(src_folder + '*.binvox')
    #dst_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/trainval_64_thin/'
    dst_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/voxel/test_thin/aug/'
    for src_model_name in src_models:
        dst_model = src_model_name.replace(src_folder, dst_folder)
        if os.path.exists(dst_model):
            continue
        else:
            print(src_model_name)
            os.system("./thinvox " + src_model_name)
            shutil.move('thinned.binvox', dst_model)


def sample_obj(phase):
    if phase == 'train':
        obj_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/trainval_rot_scl/'
        sample_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/ply/trainval/'
    else:
        obj_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/obj/test/0913/'
        sample_folder = '/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/ply/test/0913/'
    obj_list = glob.glob(obj_folder + '*.obj')
    obj_idx = 0
    for obj_name in obj_list:
        obj_idx += 1
        print('{:}/{:}'.format(obj_idx, len(obj_list)))
        ply_name = obj_name.replace(obj_folder, sample_folder)
        #ply_name = ply_name[:-4] + '.ply'
        if not os.path.exists(ply_name):
            os.system('meshlabserver -i '+ obj_name + ' -o ' + ply_name + ' -s stratified_triangle_sampling.mlx')


if __name__ == '__main__':
    #binvox()
    # surface_sample_supply()
    #thinvox()
    #sample_obj('train')
    #sample_obj('test')
    #convert_vox_to_points()



