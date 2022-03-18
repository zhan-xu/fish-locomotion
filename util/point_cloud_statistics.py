import glob
from util.mesh_obj import Mesh_obj
import numpy as np
import random

objs = glob.glob('/mnt/gypsum/mnt/nfs/work1/kalo/zhanxu/shark_pose_dataset/3d_data/ply/trainval/*.obj')
random.shuffle(objs)
for obj in objs:
    model = Mesh_obj(obj)
    print(obj)
    n1_array = []
    n2_array = []
    n3_array = []
    n4_array = []
    n5_array = []
    for ver in model.v:
        dist_list = np.sqrt(np.sum((model.v - ver)**2, axis=1))
        n1 = np.sum(dist_list < 0.06)
        n1_array.append(n1)
        n2 = np.sum(dist_list < 0.09)
        n2_array.append(n2)
        n3 = np.sum(dist_list < 0.12)
        n3_array.append(n3)
        n4 = np.sum(dist_list < 0.17)
        n4_array.append(n4)
        n5 = np.sum(dist_list < 0.2)
        n5_array.append(n5)

    print(np.mean(n1_array))
    print(np.mean(n2_array))
    print(np.mean(n3_array))
    print(np.mean(n4_array))
    print(np.mean(n5_array))