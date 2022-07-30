import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from utils.draw_utils import draw_anomaly_score_curve
from utils.pose_seg_dataset import HUMAN_IRRELATED_CLIPS


def cal_clip_roc_auc(gt_arr, scores_arr):
    auc_arr = []
    for gt, scores in zip(gt_arr, scores_arr):
        try:
            auc_arr.append(roc_auc_score(gt, scores))
        except ValueError:
            auc_arr.append(0)
    return auc_arr


def normalize_scores(score_arrs):
    score_arrs_normalized = []
    for scores in score_arrs:
        score_max = np.max(scores, axis=0)
        score_normalized = (scores / score_max) * 0.95
        score_arrs_normalized.append(score_normalized)
    return score_arrs_normalized


def smooth_scores(score_arrs, sigma=40):
    score_arrs_smoothed = []
    for scores in score_arrs:
        scores_smoothed = gaussian_filter1d(scores, sigma)
        score_arrs_smoothed.append(scores_smoothed)
    return score_arrs_smoothed


def get_dataset_scores(scores, metadata, person_keys, max_clip=None, scene_id=None, args=None):
    """
    :param scores: [samples, T]
    :param metadata: [samples, 5] scene_id, clip_id, person_id, start_frame_idx, end_frame_idx, x_g, y_g, w, h
    :param max_clip:
    :param scene_id:
    :param args:
    :return:
    """
    dataset_gt_arr = []
    dataset_scores_arr = []
    dataset_metadata_arr = []
    dataset_score_ids_arr = []
    metadata_np = np.array(metadata)
    per_frame_scores_root = f'{args.data_dir}/pose/testing/test_frame_mask/'
    clip_list = os.listdir(per_frame_scores_root)
    clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))
    if scene_id is not None:
        clip_list = [cl for cl in clip_list if int(cl[:2]) == scene_id]
    if max_clip:
        max_clip = min(max_clip, len(clip_list))
        clip_list = clip_list[:max_clip]
    print("Scoring {} clips".format(len(clip_list)))
    # calculate per clip
    for clip in clip_list:
        scene_id, clip_id = clip.split('.')[0].split('_')
        if args.hr and f'{scene_id}_{clip_id}' in HUMAN_IRRELATED_CLIPS:
            continue
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        if 'IITB' in args.data_dir:
            clip_gt = np.insert(clip_gt, 0, [0])
        scene_id, clip_id = int(scene_id), int(clip_id)
        # find the sample index of the clip
        clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                      (metadata_np[:, 0] == scene_id))[0]
        clip_metadata = metadata[clip_metadata_inds]
        # person_idxs set (person in clip)
        clip_fig_idxs = set([arr[2] for arr in clip_metadata])
        clip_frame_num = clip_gt.shape[0]
        scores_zeros = np.zeros(clip_frame_num)  # [clip frames, ]
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

        seg_len = args.seg_len
        # scores_per_frame = [[0] for _ in range(clip_frame_num)]
        for person_id in clip_fig_idxs:
            person_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                            (metadata_np[:, 0] == scene_id) &
                                            (metadata_np[:, 2] == person_id))[0]
            pid_segment_scores = scores[person_metadata_inds]  # [N, T ]
            # start frame index
            pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds])

            pid_segment_rec_scores = np.zeros(clip_frame_num)
            pid_segment_pre_scores = np.zeros(clip_frame_num)

            for segment_scores, start in zip(pid_segment_scores, pid_frame_inds):
                pid_segment_rec_scores[start:start + seg_len // 2] = np.max(
                    (pid_segment_rec_scores[start:start + seg_len // 2], segment_scores[:seg_len // 2]), axis=0)
                pid_segment_pre_scores[start + seg_len // 2:start + seg_len] = np.max(
                    (pid_segment_pre_scores[start + seg_len // 2:start + seg_len], segment_scores[seg_len // 2:]),
                    axis=0)
            clip_person_scores_dict[person_id] = pid_segment_rec_scores + pid_segment_pre_scores

        clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))  # [persons, frames_score]
        clip_score = np.amax(clip_ppl_score_arr, axis=0)
        fig_score_id = [list(clip_fig_idxs)[i] for i in np.argmax(clip_ppl_score_arr, axis=0)]
        dataset_gt_arr.append(clip_gt)
        dataset_scores_arr.append(clip_score)
        dataset_score_ids_arr.append(fig_score_id)
        dataset_metadata_arr.append([scene_id, clip_id])
    return dataset_gt_arr, dataset_scores_arr, dataset_score_ids_arr, dataset_metadata_arr


def score_dataset(score_vals, metadata, person_keys, max_clip=None, scene_id=None, args=None):
    score_vals = np.array(score_vals)  # [samples, ]
    gt_arr, scores_arr, score_ids_arr, metadata_arr = get_dataset_scores(score_vals, metadata,
                                                                         person_keys, max_clip,
                                                                         scene_id, args)
    # normalize and draw score curve
    normalized_scores = normalize_scores(scores_arr)
    # smooth
    normalized_and_smooth_scores = smooth_scores(normalized_scores, args.sigma)

    if args.vis_output:
        draw_anomaly_score_curve(normalized_scores, metadata_arr, gt_arr,
                                 cal_clip_roc_auc(gt_arr, normalized_and_smooth_scores), args.ckpt_dir.split('/')[2])

    # auc calculate
    gt_np = np.concatenate(gt_arr)
    scores_np = np.concatenate(scores_arr)
    auc, shift, sigma = score_align(scores_np, gt_np, sigma=args.sigma)
    return auc, shift, sigma


def score_align(scores_np, gt, seg_len=12, sigma=40):
    shift = seg_len + (seg_len // 2) - 1
    # scores_shifted[shift:] = scores_np[:-shift]

    scores_smoothed = gaussian_filter1d(scores_np, sigma)
    auc = roc_auc_score(gt, scores_smoothed)
    return auc, shift, sigma

