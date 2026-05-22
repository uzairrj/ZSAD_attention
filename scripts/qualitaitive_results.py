from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbones.DINO import DINOImageEncoder
from datasets import get_data
from datasets.constants import DatasetConstants
from model.model import ZSADModel
from pipeline.test import inference_epoch
from utils.args import Args
from utils.transformations import get_transforms
from utils.utils import fast_auc_and_best_f1, generate_clip_text_embeddings
from utils.visualize import visualize
from tqdm import tqdm


DEFAULT_ARGS = {
    "model_id": "openai/clip-vit-large-patch14-336",
    "vision_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vision_layers": [6, 12, 18, 24],
    "device": "cuda:0",
    "base_dir": "./",
    "mode": "test",
    "batch_size": 32,
    "img_size": 768,
    "lr": 1e-4,
    "start_epochs": 0,
    "end_epochs": 1,
    "output_dir": "./checkpoints",
    "out_dim": 768,
    "global_topk_ratio": 0.01,
}
    

def load_model_args(cli_args):
    values = dict(DEFAULT_ARGS)
    args_path = Path(cli_args.checkpoint).parent / "args.json"

    if args_path.exists():
        with args_path.open("r") as handle:
            values.update(json.load(handle))

    values.update(
        {
            "dataset_name": cli_args.dataset_name,
            "device": cli_args.device,
            "batch_size": cli_args.batch_size,
            "mode": "test",
        }
    )
    return Args(**values)


def read_images(paths):
    images = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        images.append(image)
    return images


def make_names(paths, class_names):
    fault_names = []
    img_names = []
    for index, (path, class_name) in enumerate(zip(paths, class_names)):
        image_path = Path(path)
        fault_names.append(image_path.parent.name)
        img_names.append(image_path.stem)
    return img_names, fault_names


def main():
    parser = argparse.ArgumentParser(description="Save qualitative anomaly-map results.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--darkness", type=float, default=0.6)
    parser.add_argument("--out_size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--batch_size", type=int, default=32)
    cli_args = parser.parse_args()

    checkpoint = Path(cli_args.checkpoint)

    args = load_model_args(cli_args)

    dataset_constants = DatasetConstants(args.base_dir, args.dataset_name)
    text_embeddings = generate_clip_text_embeddings(args, dataset_constants)

    transform_img, transform_mask = get_transforms(args.img_size)
    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = ZSADModel(args).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    model.eval()
    image_encoder = DINOImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)

    with torch.no_grad():
        outputs = inference_epoch(dataloader, model, image_encoder, text_embeddings)

    gt_mask_list = outputs["gt_masks"]
    pred_mask_list = outputs["pred_masks"]
    cls_list = outputs["cls_names"]
    img_paths = outputs["img_paths"]
    

    for cls_name in tqdm(np.unique(cls_list)):
        idx = (cls_list == cls_name)

        gt_cls = gt_mask_list[idx].ravel().astype(np.uint8, copy=False)
        pred_cls = pred_mask_list[idx].ravel().astype(np.float32, copy=False)

        _, best_f1, score_threshold = fast_auc_and_best_f1(gt_cls, pred_cls)

        images = read_images(img_paths[idx])
        img_names, fault_names = make_names(img_paths[idx], cls_list[idx])
        visualize(
            save_folder=cli_args.save_folder,
            names=img_names,
            object_name=cls_name,
            fault_types=fault_names,
            imgs=images,
            scores_=pred_cls.reshape(-1, *gt_mask_list.shape[1:]),
            gts=gt_cls.reshape(-1, *gt_mask_list.shape[1:]),
            alpha=cli_args.alpha,
            darkness=cli_args.darkness,
            out_size=tuple(cli_args.out_size),
            best_threshold=score_threshold,
            normalize=True,
            group_normalize=False
        )

        print(f"{cls_name}: Best pixel F1={best_f1:.5f}, Best raw score threshold={score_threshold:.5f}")


if __name__ == "__main__":
    main()
