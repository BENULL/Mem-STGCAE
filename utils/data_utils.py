import torch


def normalize_pose(pose_data, segs_meta=None, **kwargs):
    """
    Normalize keypoint values by bounding box
    :param pose_data: Formatted as [N, C, T, V], e.g. (Batch=64, Frames=12, 18, 3)
    :return:
    """
    # N, C, T, V to N, T, V, C
    pose_xy_local = pose_data.clone()
    pose_xy_local = pose_xy_local.permute(0, 2, 3, 1)

    max_kp_xy = torch.max(pose_xy_local[..., :2], dim=2)[0]
    min_kp_xy = torch.min(pose_xy_local[..., :2], dim=2)[0]

    xy_global = (max_kp_xy + min_kp_xy) / 2
    bounding_box_wh = max_kp_xy - min_kp_xy

    pose_xy_local[..., :2] = (pose_xy_local[..., :2] - xy_global[:, :, None, :]) / bounding_box_wh[:, :, None, :]

    pose_xy_perceptual = pose_data.clone()
    pose_xy_perceptual = pose_xy_perceptual.permute(0, 2, 3, 1)
    pose_xy_perceptual[..., :2] = (pose_xy_perceptual[..., :2] - min_kp_xy[:, :, None, :]) / bounding_box_wh[:, :, None, :]

    pose_xy_local = pose_xy_local.permute(0, 3, 1, 2)
    pose_xy_perceptual = pose_xy_perceptual.permute(0, 3, 1, 2)

    return pose_xy_local, pose_xy_perceptual, xy_global, bounding_box_wh


def re_normalize_pose(local_pose, normalized_pose, xy_global, bounding_box_wh):
    """
    :param normalized_pose: [N, C, T, V]
    :return:
    """
    re_normalized_pose = normalized_pose.clone()
    re_normalized_pose = re_normalized_pose.permute(0, 2, 3, 1)
    re_normalized_local_pose = local_pose.clone()
    re_normalized_local_pose = re_normalized_local_pose.permute(0, 2, 3, 1)

    min_kp_xy = (2 * xy_global - bounding_box_wh) / 2

    re_normalized_local_pose[..., :2] = re_normalized_local_pose[..., :2] * bounding_box_wh[:, :, None, :] + xy_global[
                                                                                                             :, :, None,
                                                                                                             :]
    re_normalized_pose[..., :2] = re_normalized_pose[..., :2] * bounding_box_wh[:, :, None, :] + min_kp_xy[:, :, None,
                                                                                                 :]

    re_normalized_pose = re_normalized_pose.permute(0, 3, 1, 2)
    re_normalized_local_pose = re_normalized_local_pose.permute(0, 3, 1, 2)

    return re_normalized_local_pose, re_normalized_pose
