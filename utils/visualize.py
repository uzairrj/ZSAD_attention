import os

import cv2
import numpy as np

def make_single_result(image, gt, score, alpha=0.5, darkness=0.6,
                       out_size=(512, 512), threshold=0.5, min=None, max=None):
    h, w = image.shape[:2]

    if gt is None:
        gt = np.zeros((h, w), dtype=np.uint8)
    else:
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    score = cv2.resize(score, (w, h), interpolation=cv2.INTER_CUBIC)

    if min is not None and max is not None:
        score = (score - min) / (max - min + 1e-8)
        threshold = (threshold - min) / (max - min)
        threshold = np.clip(threshold, 0.0, 1.0)

    gt_mask = gt > 0
    darkness = np.clip(darkness, 0.0, 1.0)
    gt_image = (image * (1.0 - darkness)).astype(np.uint8)
    gt_image[gt_mask] = image[gt_mask]

    score = np.clip(score, 0.0, 1.0)
    score_u8 = (score * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_u8, cv2.COLORMAP_JET)
    pred_image = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)

    threshold_value = np.clip(threshold, 0.0, 1.0)
    binary_map = score >= threshold_value
    threshold_score = score * binary_map
    heatmap_threshold = cv2.applyColorMap((np.clip(threshold_score, 0.0, 1.0)*255).astype(np.uint8), cv2.COLORMAP_JET)
    pred_image_threshold = image.copy()
    pred_image_threshold = cv2.addWeighted(image, 1.0 - alpha, heatmap_threshold, alpha, 0)

    out_h, out_w = out_size
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    gt_image = cv2.resize(gt_image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    pred_image = cv2.resize(pred_image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    pred_image_threshold = cv2.resize(pred_image_threshold, (out_w, out_h), interpolation=cv2.INTER_AREA)
    result = np.concatenate([image, gt_image, pred_image, pred_image_threshold], axis=1)

    return result


def visualize(save_folder,names, object_name, fault_types, imgs, scores_, gts,
                            alpha=0.5, darkness=0.6, out_size=(512, 512), best_threshold=0.5, normalize=False, group_normalize=False):
    os.makedirs(save_folder, exist_ok=True)

    if isinstance(scores_, list):
        scores_ = np.array(scores_)

    total_number = len(imgs)
    score_min = None
    score_max = None

    if normalize and group_normalize:
        score_min = scores_.min()
        score_max = scores_.max()

    for idx in range(total_number):
        if normalize and not group_normalize:
            score_min = scores_.min()
            score_max = scores_.max()
        result = make_single_result(
            imgs[idx],
            gts[idx],
            scores_[idx],
            alpha=alpha,
            darkness=darkness,
            out_size=out_size,
            threshold=best_threshold,
            min=score_min,
            max=score_max
        )

        save_name = f"{names[idx]}.png"

        complete_save_path = os.path.join(save_folder, object_name, fault_types[idx])

        os.makedirs(complete_save_path, exist_ok=True)
        cv2.imwrite(os.path.join(complete_save_path, save_name), result)