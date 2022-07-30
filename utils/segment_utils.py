import numpy as np


def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        # Meta is [person_id, first_frame]
        sing_pose_np, sing_pose_meta, sing_pose_keys = single_pose_dict2np(clip_dict, idx)
        key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys

        curr_pose_segs_np, curr_pose_segs_meta = split_pose_to_segments(sing_pose_np, sing_pose_meta, sing_pose_keys,
                                                                        start_ofst, seg_stride, seg_len,
                                                                        scene_id=scene_id, clip_id=clip_id)
        pose_segs_data.append(curr_pose_segs_np)
        pose_segs_meta += curr_pose_segs_meta
    pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)

    del pose_segs_data
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys
    else:
        return pose_segs_data_np, pose_segs_meta


def single_pose_dict2np(person_dict, idx):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys


def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id=''):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            end_key = single_pose_keys_sorted[start_ind + seg_len - 1]
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key), int(end_key)])
    return pose_segs_np, pose_segs_meta




