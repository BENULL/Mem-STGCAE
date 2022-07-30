#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2022/2/22 下午6:40
"""
import cv2
import os
import shutil
import numpy as np
import time
from matplotlib import pyplot as plt
from utils.data_utils import re_normalize_pose

_OPENPOSE_POINT_COLORS = [
    (255, 255, 0), (255, 191, 0),
    (102, 255, 0), (255, 77, 0), (0, 255, 0),
    (255, 255, 77), (204, 255, 77), (255, 204, 77),
    (77, 255, 191), (255, 191, 77), (77, 255, 91),
    (255, 77, 204), (204, 255, 77), (255, 77, 191),
    (191, 255, 77), (255, 77, 127),
    (127, 255, 77), (255, 255, 0)]

_OPENPOSE_EDGES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16),
    (0, 15), (15, 17)
]

_OPENPOSE_EDGE_COLORS = [
    (0, 0, 255),
    (0, 84, 255), (0, 168, 0), (0, 255, 168),
    (84, 0, 168), (84, 84, 255), (84, 168, 0),
    (84, 255, 84), (168, 0, 255), (168, 84, 255),
    (168, 168, 0), (168, 255, 84), (255, 0, 0),
    (255, 84, 255), (255, 168, 0),
    (255, 255, 84), (255, 0, 168)]

_OPENPOSE_EDGE_COLORS_BLUE = [(237, 40, 33)] * 17
_OPENPOSE_EDGE_COLORS_RED = [(0, 0, 255)]*17
_OPENPOSE_EDGE_COLORS_YELLOW = [(0, 255, 255)]*17
_OPENPOSE_POINT_COLORS_BLUE = [(255, 9, 9)] * 18
_OPENPOSE_POINT_COLORS_RED = [(0, 0, 255)]*18
_OPENPOSE_POINT_COLORS_YELLOW = [(0, 255, 255)]*18


RENDER_CONFIG_OPENPOSE = {
    'edges': _OPENPOSE_EDGES,
    'edgeColors': _OPENPOSE_EDGE_COLORS_BLUE,
    'edgeWidth': 1,
    'pointColors': _OPENPOSE_POINT_COLORS_BLUE,
    'pointRadius': 2
}


def preparePoint(points, imageSize, invNorm):
    if invNorm == 'auto':
        invNorm = np.bitwise_and(points >= 0, points <= 1).all()

    if invNorm:
        w, h = imageSize
        trans = np.array([[w, 0], [0, h]])
        points = (trans @ points.T).T

    return points.astype(np.int32)


def renderPose(image, poses, inplace: bool = True, inverseNormalization='auto'):
    """绘制骨架

    参数
        image: 原图

        poses: 一组或多组关节点坐标

        config: 配置项

        inplace: 是否绘制在原图上

        inverseNormalization` 是否[True|False]进行逆归一化, 当值为auto时将根据坐标值自动确定

    返回
        输出图像, inplace为True时返回image, 为False时返回新的图像
    """
    poses = np.array(poses)
    if not inplace:
        image = image.copy()

    if len(poses.shape) == 2:
        poses = poses[None, :, :2]

    if inverseNormalization not in ['auto', True, False]:
        raise ValueError(f'Unknown "inverseNormalization" value {inverseNormalization}')

    _isPointValid = lambda point: point[0] != 0 and point[1] != 0
    _FILL_CIRCLE = -1
    for pose in poses:
        pose = preparePoint(pose, (image.shape[1], image.shape[0]), inverseNormalization)
        validPointIndices = set(filter(lambda i: _isPointValid(pose[i]), range(pose.shape[0])))
        for i, (start, end) in enumerate(RENDER_CONFIG_OPENPOSE['edges']):
            if start in validPointIndices and end in validPointIndices:
                cv2.line(image, tuple(pose[start]), tuple(pose[end]), RENDER_CONFIG_OPENPOSE['edgeColors'][i],
                         RENDER_CONFIG_OPENPOSE['edgeWidth'])

        for i in validPointIndices:
            cv2.circle(image, tuple(pose[i]), RENDER_CONFIG_OPENPOSE['pointRadius'],
                       tuple(RENDER_CONFIG_OPENPOSE['pointColors'][i]), _FILL_CIRCLE)

    return image


def renderBbox(image, box, inplace: bool = True, inverseNormalization='auto'):
    if not inplace:
        image = image.copy()

    if inverseNormalization not in ['auto', True, False]:
        raise ValueError(f'Unknown "inverseNormalization" value {inverseNormalization}')
    if len(box) == 4:
        box = np.array(box).reshape(2, 2)
        box = preparePoint(box, (image.shape[1], image.shape[0]), inverseNormalization)
        cv2.rectangle(image, tuple(box[0]), tuple(box[0]+box[1]), (0, 0, 255), thickness=2)
    return image

def draw_predict_skeleton( metas, output_arr, date_time, v_id):

    RENDER_CONFIG_OPENPOSE['pointColors'] = _OPENPOSE_POINT_COLORS_RED
    RENDER_CONFIG_OPENPOSE['edgeColors'] = _OPENPOSE_EDGE_COLORS_RED

    video_path = f'/root/VAD/lvad/visualization/{v_id}/'
    visualization_path = f'/root/VAD/lvad/visualization/{date_time}_{v_id}_res/'
    scene_id, clip_id = v_id.split('_')
    if not os.path.exists(visualization_path):
        shutil.copytree(video_path, visualization_path)

    for predict, meta in zip(output_arr, metas):
        if meta[0] == int(scene_id) and meta[1] == int(clip_id):
            img_idx = meta[4]
            # DRAW
            img_path = visualization_path + f'{img_idx:03d}.jpg'
            img = cv2.imread(img_path)

            # renormalize
            predict = predict.squeeze(1).T
            predict = re_normalize_pose(predict, meta[5:])

            renderPoseImage = renderPose(img, predict, inplace=False)
            cv2.imwrite(img_path, renderPoseImage)


def draw_mask_skeleton(targets, predicts, metas, dir):
    dir = f'/root/VAD/lvad/visualization/mask/{dir}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for target_b, predict_b, meta in zip(targets, predicts, metas):
        scene_id, clip_id, person_id, start_frame = meta[:4]
        if scene_id not in [1] or clip_id not in [14]:
            continue

        mask_imgs = []
        target_b = np.transpose(target_b, (1,2,0))
        predict_b = np.transpose(predict_b, (1,2,0))

        frame = 1
        for target, predict in zip(target_b, predict_b):
            mask_img = np.zeros((480, 856, 3), np.uint8)
            mask_img.fill(255)

            RENDER_CONFIG_OPENPOSE['pointColors'] = _OPENPOSE_POINT_COLORS_BLUE
            RENDER_CONFIG_OPENPOSE['edgeColors'] = _OPENPOSE_EDGE_COLORS_BLUE
            mask_img = renderPose(mask_img, target, inplace=False)

            if frame >= 7:
                RENDER_CONFIG_OPENPOSE['pointColors'] = _OPENPOSE_POINT_COLORS_YELLOW
                RENDER_CONFIG_OPENPOSE['edgeColors'] = _OPENPOSE_EDGE_COLORS_YELLOW
                mask_img = renderPose(mask_img, predict, inplace=False)
            else:
                RENDER_CONFIG_OPENPOSE['pointColors'] = _OPENPOSE_POINT_COLORS_RED
                RENDER_CONFIG_OPENPOSE['edgeColors'] = _OPENPOSE_EDGE_COLORS_RED
                mask_img = renderPose(mask_img, predict, inplace=False)

            frame += 1
            mask_imgs.append(mask_img)


        mask_imgs = np.concatenate(mask_imgs, axis=0)
        cv2.imwrite(f'{dir}/{scene_id}-{clip_id}-{person_id}-{start_frame}-{int(time.time()*10000)}.jpg', mask_imgs)


def draw_anomaly_score_curve(score_arrs, meta_arrs, gt_arrs, aucs, dir):
    dir = f'/root/VAD/lvad/visualization/score_curve/{dir}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for scores, meta, gt_arr, auc in zip(score_arrs, meta_arrs, gt_arrs, aucs):
        scene = f'{meta[0]:02d}_{meta[1]:04d}'
        fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Anomaly Score')
        ax.text(-0.1, 0.95, f'Scene: {scene}')
        ax.text(-0.1, 0.85, f'AUC: {auc:.4f}')
        ax.set_yticks([0, 1])
        ax.plot(np.arange(len(scores)), gt_arr, color='r', zorder=2)
        ax.plot(np.arange(len(scores)), scores, color='b', zorder=1)
        plt.savefig(f'{dir}/{scene}_{int(auc*10000)}_{int(time.time())}.png')
        plt.close()
