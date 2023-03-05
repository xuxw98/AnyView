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

    points = np.transpose(np.vstack((position[0:3, :], R, G, B, valid.nonzero()[0])))
    return points

if __name__ == '__main__':
    f = open('meta_data/scannet_train.txt', 'r')
    unify_dim = (320, 240)
    unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
    scene_names = f.readlines()
    for scene_name in scene_names:
        scene_name = scene_name[:-1]
        num_frames = len(os.listdir('2D/%s/color/' % scene_name))
        # Load scene axis alignment matrix
        lines = open('scans/%s/%s.txt' % (scene_name, scene_name)).readlines()
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                    for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        sum_views = 0
        imgs = []
        pcs = []
        poses = []
        masks = []
        boxes = np.load('scannet_train_detection_data/'+scene_name+'_bbox.npy')
        for i in range(num_frames):
            frame_id = i * 20
            f = os.path.join(path_2d, scene_name, 'color', str(frame_id)+'.jpg')
            img = imageio.imread(f)
            depth = imageio.imread(f.replace('color', 'depth').replace('jpg', 'png')) / 1000.0  # convert to meter
            posePath = f.replace('color', 'pose').replace('.jpg', '.txt')
            pose = np.asarray(
                [[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
                (x.split(" ") for x in open(posePath).read().splitlines())]
            )
            pc = depth_image_to_point_cloud(img, depth, unify_intrinsic[:3,:3], pose)
            if np.isnan(pc).any():
                continue
            try:
                pc = pc[np.random.choice(pc.shape[0], 5000, replace=False)]
            except:
                continue
            sum_views += 1
            pts = np.ones((pc.shape[0], 4))
            pts[:,0:3] = pc[:,0:3]
            pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
            pc[:,0:3] = pts[:,0:3]
            pose_ = np.dot(axis_align_matrix, pose)
            imgs.append(img)
            pcs.append(pc)
            poses.append(pose_)

            # Box Mask
            box_center = boxes[:,:3].copy()
            box_center -= pose_[:3,3]
            box_center = np.dot(pose_[:3,:3].transpose(), box_center.transpose()).transpose() # Nx3
            uv = np.dot(box_center, unify_intrinsic[:3, :3].transpose())
            uv[:,0] /= uv[:,2]
            uv[:,1] /= uv[:,2]
            masks.append((uv[:,0]>=0) * (uv[:,0]<320) * (uv[:,1]>=0) * (uv[:,1]<240) * (box_center[:,2]>0))

        imgs = np.stack(imgs, axis=0)
        pcs = np.stack(pcs, axis=0)
        poses = np.stack(poses, axis=0)
        masks = np.stack(masks, axis=0).astype(np.float32)
        # Sample 50 views, 50 is the maximum view number.
        if sum_views > 50:
            delta = (sum_views - 1) / 49.0
            ids = []
            for i in range(50):
                ids.append(int(delta * i))
            imgs, pcs, poses = imgs[ids], pcs[ids], poses[ids]
        print(scene_name, min(sum_views, 50))
        np.savez_compressed('anyview_2d_data/' + scene_name + '_data.npz', imgs=imgs, pcs=pcs, poses=poses, masks=masks)