from tqdm import tqdm
import os, struct
import numpy as np
import zlib
import imageio
import cv2
import math
import sys

path_2d = './2D/'

# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def depth_image_to_point_cloud(rgb, depth, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])
    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float)
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B)))
    return points

if __name__ == '__main__':
    f = open('meta_data/scannet_train.txt', 'r')
    num_views = 10
    scene_names = f.readlines()
    for scene_name in scene_names:
        scene_name = scene_name[:-1]
        num_frames = len(os.listdir('2D/%s/color/' % scene_name))
        pcs = []
        sum_views = 0
        if num_frames <= num_views:
            num_iter = num_frames
        else:
            num_iter = num_views
        delta = (num_frames - 1.0) / (num_iter - 1)
        for i in range(num_iter):
            frame_id = int(delta * i) * 20
            f = os.path.join(path_2d, scene_name, 'color', str(frame_id)+'.jpg')
            img = imageio.imread(f)
            depth = imageio.imread(f.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter
            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                (x.split(" ") for x in open(posePath).read().splitlines())]
            )
            unify_dim = (320, 240)
            img = cv2.resize(img, unify_dim, interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, unify_dim, interpolation=cv2.INTER_NEAREST)
            unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
            pc = depth_image_to_point_cloud(img, depth, unify_intrinsic[:3,:3], pose)
            if np.isnan(pc).any():
                continue
            #print(pc.shape[0])
            #np.savetxt('depthes/%d.txt' % frame_id, pc)
            pcs.append(pc)
            sum_views += 1
        pcs = np.concatenate(pcs, axis=0)
        print(scene_name, sum_views, pcs.shape)
        pcs = pcs[np.random.choice(pcs.shape[0], 50000, replace=False)]
        # Load scene axis alignment matrix
        lines = open('scans/%s/%s.txt' % (scene_name, scene_name)).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        pts = np.ones((pcs.shape[0], 4))
        pts[:,0:3] = pcs[:,0:3]
        pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
        pcs[:,0:3] = pts[:,0:3]
        np.save('pc_fusion_%dview/' % num_views + scene_name + '_vert.npy', pcs)
