import argparse
import os

import cv2
import numpy as np


def read_score(path):
    if path.endswith(".npy"):
        return np.load(path)

    if path.endswith(".npz"):
        data = np.load(path)
        key = "img" if "img" in data.files else data.files[0]
        return data[key]

    score = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if score is None:
        raise FileNotFoundError(path)
    return score


def normalize_score(score, min_value=None, max_value=None):
    score = score.astype(np.float32)
    if min_value is None:
        min_value = np.min(score)
    if max_value is None:
        max_value = np.max(score)

    if max_value - min_value < 1e-8:
        return np.zeros_like(score, dtype=np.uint8)

    score = (score - min_value) / (max_value - min_value)
    score = np.clip(score, 0.0, 1.0)
    return (score * 255).astype(np.uint8)


def make_single_result(image, gt, score, alpha=0.5, darkness=0.6,
                       out_size=(512, 512), threshold=0.5):
    h, w = image.shape[:2]

    if gt is None:
        gt = np.zeros((h, w), dtype=np.uint8)
    else:
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

    score = cv2.resize(score, (w, h), interpolation=cv2.INTER_CUBIC)

    gt_mask = gt > 0
    darkness = np.clip(darkness, 0.0, 1.0)
    gt_image = (image * (1.0 - darkness)).astype(np.uint8)
    gt_image[gt_mask] = image[gt_mask]

    heatmap = cv2.applyColorMap(score, cv2.COLORMAP_JET)
    pred_image = cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0)

    binary_map = score > int(threshold * 255)
    effect_area = score * binary_map.astype(np.uint8)
    heatmap_threshold = cv2.applyColorMap(effect_area, cv2.COLORMAP_JET)
    pred_image_threshold = cv2.addWeighted(image, 1.0 - alpha, heatmap_threshold, alpha, 0)

    out_h, out_w = out_size
    image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    gt_image = cv2.resize(gt_image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    pred_image = cv2.resize(pred_image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    pred_image_threshold = cv2.resize(pred_image_threshold, (out_w, out_h), interpolation=cv2.INTER_AREA)
    result = np.concatenate([image, gt_image, pred_image, pred_image_threshold], axis=1)

    return result


def visualize(names, imgs, scores_, gts, save_folder,
                            alpha=0.5, darkness=0.6, out_size=(512, 512), best_threshold=0.5, global_normalization=False):
    os.makedirs(save_folder, exist_ok=True)

    if not isinstance(scores_, dict):
        scores_ = {"": scores_}

    total_number = len(imgs)
    for key, score_list in scores_.items():
        if len(score_list) != total_number:
            raise ValueError(f"Score count for {key} does not match image count")

    for key, score_list in scores_.items():
        min_value = min(np.min(score) for score in score_list)
        max_value = max(np.max(score) for score in score_list)

        if not global_normalization:
            min_value = None
            max_value = None

        for idx in range(total_number):
            score = normalize_score(score_list[idx],min_value=min_value, max_value=max_value)
            result = make_single_result(
                imgs[idx],
                gts[idx],
                score,
                alpha=alpha,
                darkness=darkness,
                out_size=out_size,
                threshold=best_threshold,
            )

            if key == "":
                save_name = f"{names[idx]}.png"
            else:
                save_name = f"{names[idx]}_{key}.png"
            cv2.imwrite(os.path.join(save_folder, save_name), result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--score", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--darkness", type=float, default=0.6)
    parser.add_argument("--out-size", type=int, nargs=2, default=(512, 512),
                        metavar=("H", "W"))
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    gt = None
    if args.gt is not None and args.gt != "0":
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(args.gt)

    score = read_score(args.score)
    score = normalize_score(score)
    result = make_single_result(
        image,
        gt,
        score,
        alpha=args.alpha,
        darkness=args.darkness,
        out_size=tuple(args.out_size),
    )
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.out, result)


if __name__ == "__main__":
    main()
