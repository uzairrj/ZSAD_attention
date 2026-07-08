from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLIPTextEncoder = None
DINOImageEncoder = None
DatasetConstants = None
ZSADModel = None
Args = None
get_data = None
get_transforms = None
prompt_generator = None


def load_project_modules() -> None:
    global CLIPTextEncoder
    global DINOImageEncoder
    global DatasetConstants
    global ZSADModel
    global Args
    global get_data
    global get_transforms
    global prompt_generator

    from backbones.CLIP import CLIPTextEncoder as _CLIPTextEncoder
    from backbones.DINO import DINOImageEncoder as _DINOImageEncoder
    from datasets import get_data as _get_data
    from datasets.constants import DatasetConstants as _DatasetConstants
    from model.model import ZSADModel as _ZSADModel
    from utils.args import Args as _Args
    from utils.transformations import get_transforms as _get_transforms
    from utils.utils import prompt_generator as _prompt_generator

    CLIPTextEncoder = _CLIPTextEncoder
    DINOImageEncoder = _DINOImageEncoder
    DatasetConstants = _DatasetConstants
    ZSADModel = _ZSADModel
    Args = _Args
    get_data = _get_data
    get_transforms = _get_transforms
    prompt_generator = _prompt_generator


DEFAULTS = {
    "model_id": "openai/clip-vit-large-patch14-336",
    "vision_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vision_layers": [6, 12, 18, 24],
    "device": "cuda:0",
    "base_dir": "./",
    "dataset_name": "mvtec",
    "mode": "test",
    "batch_size": 1,
    "img_size": 768,
    "lr": 1e-4,
    "start_epochs": 0,
    "end_epochs": 1,
    "output_dir": "./checkpoints_mvtec",
    "out_dim": 768,
    "global_topk_ratio": 0.01,
}


def parse_vision_layers(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "sample"


def load_json(path: Path) -> Dict:
    with path.open("r") as handle:
        return json.load(handle)


def checkpoint_args_path(checkpoint: Path, explicit_path: Optional[str]) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    sibling = checkpoint.parent / "args.json"
    return sibling if sibling.exists() else None


def build_model_args(cli_args: argparse.Namespace) -> Args:
    values = dict(DEFAULTS)
    checkpoint = Path(cli_args.checkpoint)

    args_path = checkpoint_args_path(checkpoint, cli_args.checkpoint_args)
    if args_path is not None:
        values.update(load_json(args_path))

    values.update(
        {
            "device": cli_args.device,
            "dataset_name": cli_args.dataset_name,
            "img_size": cli_args.img_size,
            "base_dir": cli_args.base_dir,
            "model_id": cli_args.model_id,
            "vision_model_id": cli_args.vision_model_id,
            "vision_layers": cli_args.vision_layers,
            "out_dim": cli_args.out_dim,
            "global_topk_ratio": cli_args.global_topk_ratio,
        }
    )
    return Args(**values)


def load_checkpoint(path: Path, device: str) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict in checkpoint: {path}")

    if any(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def maybe_use_cpu(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is not available; falling back from {device} to cpu.")
        return "cpu"
    return device


def encode_prompt_groups(
    args: Args,
    prompts: Dict[str, Dict[str, List[str]]],
    class_name: str,
    text_encoder,
    prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, List[str]]]],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List[str]]]:
    if class_name in prompt_cache:
        return prompt_cache[class_name]

    if class_name not in prompts["normal"]:
        known = ", ".join(sorted(prompts["normal"].keys()))
        raise KeyError(f"Unknown class '{class_name}' for {args.dataset_name}. Known classes: {known}")

    normal_text = prompts["normal"][class_name]
    abnormal_text = prompts["abnormal"][class_name]
    normal_embeddings = text_encoder(normal_text).unsqueeze(0)
    abnormal_embeddings = text_encoder(abnormal_text).unsqueeze(0)
    encoded = (normal_embeddings, abnormal_embeddings, {"normal": normal_text, "abnormal": abnormal_text})
    prompt_cache[class_name] = encoded
    return encoded


def adapt_prompt_embeddings(
    model: ZSADModel,
    normal_embeddings: torch.Tensor,
    abnormal_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = model.positive_text_adapter(normal_embeddings)
    positive = model.normalize_features(positive)
    negative = model.negative_text_adapter(abnormal_embeddings)
    negative = model.normalize_features(negative)
    return positive, negative, model.merge_text_embeddings(positive, negative)


def patch_grid_size(patch_count: int) -> int:
    grid = int(math.sqrt(patch_count))
    if grid * grid != patch_count:
        raise ValueError(f"Expected a square patch grid, got {patch_count} patches.")
    return grid


def cross_attention_weights(
    attention_module,
    patch_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    q, k, _ = attention_module._qkv(patch_embeddings, text_embeddings)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attention_module.head_dim)
    return torch.softmax(scores, dim=-1).mean(dim=1)


def group_attention_maps(
    model: ZSADModel,
    patches: torch.Tensor,
    normal_embeddings: torch.Tensor,
    abnormal_embeddings: torch.Tensor,
    aggregate: str,
) -> List[Dict[str, torch.Tensor]]:
    positive_text, negative_text, adapted_text = adapt_prompt_embeddings(
        model,
        normal_embeddings,
        abnormal_embeddings,
    )
    positive_count = positive_text.shape[1]

    layer_maps = []
    for layer_idx in range(model.number_of_vision_layers):
        patch_embeddings = model.patch_adapter[layer_idx](patches[:, layer_idx, :, :])
        patch_embeddings = model.normalize_features(patch_embeddings)

        weights = cross_attention_weights(
            model.cross_model_contrastive_learning_attention[layer_idx],
            patch_embeddings,
            adapted_text,
        )
        positive_weights = weights[:, :, :positive_count]
        negative_weights = weights[:, :, positive_count:]

        if aggregate == "sum":
            positive_scores = positive_weights.sum(dim=-1)
            negative_scores = negative_weights.sum(dim=-1)
        elif aggregate == "mean":
            positive_scores = positive_weights.mean(dim=-1)
            negative_scores = negative_weights.mean(dim=-1)
        else:
            raise ValueError(f"Unsupported aggregate mode: {aggregate}")

        grid = patch_grid_size(positive_scores.shape[1])
        layer_maps.append(
            {
                "positive": positive_scores[0].reshape(grid, grid).detach().cpu(),
                "negative": negative_scores[0].reshape(grid, grid).detach().cpu(),
            }
        )
    return layer_maps


def normalize_heatmap(heatmap: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.clip(heatmap, 0.0, 1.0)
    if mode != "per-map":
        raise ValueError(f"Unsupported heatmap normalization: {mode}")

    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        return np.zeros_like(heatmap, dtype=np.float32)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value - min_value < 1e-12:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((heatmap - min_value) / (max_value - min_value)).astype(np.float32)


def signed_attention_grid(maps: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    positive = maps["positive"]
    negative = maps["negative"]
    if mode == "difference":
        return positive - negative
    if mode == "normalized-difference":
        return (positive - negative) / (positive + negative + 1e-8)
    raise ValueError(f"Unsupported signed score mode: {mode}")


def normalize_signed_heatmap(heatmap: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.clip((heatmap + 1.0) / 2.0, 0.0, 1.0)
    if mode != "symmetric":
        raise ValueError(f"Unsupported signed heatmap normalization: {mode}")

    finite = heatmap[np.isfinite(heatmap)]
    if finite.size == 0:
        return np.full_like(heatmap, 0.5, dtype=np.float32)
    max_abs = float(np.max(np.abs(finite)))
    if max_abs < 1e-12:
        return np.full_like(heatmap, 0.5, dtype=np.float32)
    return np.clip((heatmap / max_abs + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def upsample_grid(grid: torch.Tensor, image_size: int) -> np.ndarray:
    resized = F.interpolate(
        grid[None, None].float(),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def top_patch_indices(grid: torch.Tensor, topk: int) -> List[int]:
    if topk <= 0:
        return []
    flat = grid.flatten()
    k = min(topk, flat.numel())
    return torch.topk(flat, k=k).indices.cpu().tolist()


def bottom_patch_indices(grid: torch.Tensor, topk: int) -> List[int]:
    if topk <= 0:
        return []
    flat = grid.flatten()
    k = min(topk, flat.numel())
    return torch.topk(-flat, k=k).indices.cpu().tolist()


def draw_patch_boxes(
    image: Image.Image,
    patch_indices: Sequence[int],
    grid_size: int,
    color: Tuple[int, int, int],
) -> Image.Image:
    boxed = image.copy()
    draw = ImageDraw.Draw(boxed)
    width, height = boxed.size
    line_width = max(2, width // 256)

    for patch_idx in patch_indices:
        row, col = divmod(int(patch_idx), grid_size)
        x0 = round(col * width / grid_size)
        y0 = round(row * height / grid_size)
        x1 = round((col + 1) * width / grid_size) - 1
        y1 = round((row + 1) * height / grid_size) - 1
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)

    return boxed


def make_overlay(
    display_image: Image.Image,
    score_grid: torch.Tensor,
    cmap_name: str,
    box_color: Tuple[int, int, int],
    alpha: float,
    topk: int,
    normalization: str,
) -> Image.Image:
    image_size = display_image.size[0]
    score_heatmap = upsample_grid(score_grid, image_size)
    normalized = normalize_heatmap(score_heatmap, normalization)
    color_map = plt.get_cmap(cmap_name)(normalized)[..., :3]

    base = np.asarray(display_image).astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * color_map
    blended = Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8))

    return draw_patch_boxes(
        blended,
        top_patch_indices(score_grid, topk),
        score_grid.shape[0],
        box_color,
    )


def make_signed_overlay(
    display_image: Image.Image,
    signed_grid: torch.Tensor,
    alpha: float,
    topk: int,
    normalization: str,
) -> Image.Image:
    image_size = display_image.size[0]
    score_heatmap = upsample_grid(signed_grid, image_size)
    normalized = normalize_signed_heatmap(score_heatmap, normalization)
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "negative_white_positive",
        ["#d7191c", "#f7f7f7", "#1a9641"],
    )(normalized)[..., :3]

    base = np.asarray(display_image).astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * color_map
    blended = Image.fromarray(np.clip(blended * 255.0, 0, 255).astype(np.uint8))

    boxed = draw_patch_boxes(
        blended,
        top_patch_indices(signed_grid, topk),
        signed_grid.shape[0],
        (0, 255, 120),
    )
    return draw_patch_boxes(
        boxed,
        bottom_patch_indices(signed_grid, topk),
        signed_grid.shape[0],
        (255, 40, 40),
    )


def save_attention_figure(
    output_path: Path,
    display_image: Image.Image,
    layer_maps: List[Dict[str, torch.Tensor]],
    vision_layers: Sequence[int],
    topk: int,
    alpha: float,
    normalization: str,
    signed_normalization: str,
    signed_score: str,
    view: str,
) -> None:
    rows = len(layer_maps)
    columns = 1 if view == "signed" else 2 if view == "split" else 3
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, max(3, rows * 3.2)), squeeze=False)

    for row_idx, maps in enumerate(layer_maps):
        layer_label = vision_layers[row_idx] if row_idx < len(vision_layers) else row_idx
        col_idx = 0
        if view in {"signed", "both"}:
            signed_grid = signed_attention_grid(maps, signed_score)
            overlay = make_signed_overlay(
                display_image,
                signed_grid,
                alpha=alpha,
                topk=topk,
                normalization=signed_normalization,
            )
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(
                f"DINO layer {layer_label}: positive - negative\n"
                f"red=negative, green=positive, range=[{signed_grid.min().item():.4f}, {signed_grid.max().item():.4f}]",
                fontsize=10,
            )
            col_idx += 1

        panels = []
        if view in {"split", "both"}:
            panels = [
                ("positive / normal", maps["positive"], "viridis", (0, 255, 170)),
                ("negative / abnormal", maps["negative"], "magma", (255, 70, 70)),
            ]
        for col_idx, (label, grid, cmap_name, box_color) in enumerate(panels):
            overlay = make_overlay(
                display_image,
                grid,
                cmap_name=cmap_name,
                box_color=box_color,
                alpha=alpha,
                topk=topk,
                normalization=normalization,
            )
            axis = axes[row_idx, col_idx + (1 if view == "both" else 0)]
            axis.imshow(overlay)
            axis.axis("off")
            axis.set_title(
                f"DINO layer {layer_label}: {label}\n"
                f"raw min={grid.min().item():.4f}, max={grid.max().item():.4f}",
                fontsize=10,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def layer_metadata(
    layer_maps: List[Dict[str, torch.Tensor]],
    vision_layers: Sequence[int],
    topk: int,
    signed_score: Optional[str] = None,
) -> List[Dict]:
    metadata = []
    for layer_idx, maps in enumerate(layer_maps):
        layer_label = vision_layers[layer_idx] if layer_idx < len(vision_layers) else layer_idx
        entry = {"vision_layer": int(layer_label)}
        for group_name, grid in maps.items():
            indices = top_patch_indices(grid, topk)
            patches = []
            grid_size = grid.shape[0]
            for patch_idx in indices:
                row, col = divmod(int(patch_idx), grid_size)
                patches.append(
                    {
                        "index": int(patch_idx),
                        "row": int(row),
                        "col": int(col),
                        "score": float(grid[row, col].item()),
                    }
                )
            entry[group_name] = {
                "min": float(grid.min().item()),
                "max": float(grid.max().item()),
                "mean": float(grid.mean().item()),
                "top_patches": patches,
            }
        if signed_score is not None:
            signed_grid = signed_attention_grid(maps, signed_score)
            signed_size = signed_grid.shape[0]
            positive_patches = []
            negative_patches = []
            for patch_idx in top_patch_indices(signed_grid, topk):
                row, col = divmod(int(patch_idx), signed_size)
                positive_patches.append(
                    {
                        "index": int(patch_idx),
                        "row": int(row),
                        "col": int(col),
                        "score": float(signed_grid[row, col].item()),
                    }
                )
            for patch_idx in bottom_patch_indices(signed_grid, topk):
                row, col = divmod(int(patch_idx), signed_size)
                negative_patches.append(
                    {
                        "index": int(patch_idx),
                        "row": int(row),
                        "col": int(col),
                        "score": float(signed_grid[row, col].item()),
                    }
                )
            entry["signed"] = {
                "score": signed_score,
                "min": float(signed_grid.min().item()),
                "max": float(signed_grid.max().item()),
                "mean": float(signed_grid.mean().item()),
                "top_positive_patches": positive_patches,
                "top_negative_patches": negative_patches,
            }
        metadata.append(entry)
    return metadata


def save_numpy_maps(output_path: Path, layer_maps: List[Dict[str, torch.Tensor]], signed_score: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        positive=np.stack([layer["positive"].numpy() for layer in layer_maps], axis=0),
        negative=np.stack([layer["negative"].numpy() for layer in layer_maps], axis=0),
        signed=np.stack([signed_attention_grid(layer, signed_score).numpy() for layer in layer_maps], axis=0),
    )


def load_display_image(image_path: str, image_size: int) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)


def dataset_samples(cli_args: argparse.Namespace, args: Args) -> Iterable[Tuple[int, Dict]]:
    transform_img, transform_mask = get_transforms(args.img_size)
    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=False)

    selected = 0
    for index in range(cli_args.start_index, len(dataset)):
        sample = dataset[index]
        if cli_args.class_name and sample["cls_name"] != cli_args.class_name:
            continue
        if cli_args.anomaly == "normal" and int(sample["anomaly"]) != 0:
            continue
        if cli_args.anomaly == "abnormal" and int(sample["anomaly"]) != 1:
            continue

        yield index, sample
        selected += 1
        if selected >= cli_args.num_samples:
            break


def direct_image_sample(cli_args: argparse.Namespace, args: Args) -> Tuple[int, Dict]:
    if not cli_args.class_name:
        raise ValueError("--class-name is required when --image-path is used.")
    transform_img, _ = get_transforms(args.img_size)
    image = Image.open(cli_args.image_path).convert("RGB")
    return 0, {
        "img": transform_img(image),
        "img_mask": None,
        "cls_name": cli_args.class_name,
        "anomaly": -1,
        "img_path": cli_args.image_path,
    }


def process_sample(
    sample_index: int,
    sample: Dict,
    args: Args,
    model: ZSADModel,
    image_encoder: DINOImageEncoder,
    prompts: Dict[str, Dict[str, List[str]]],
    text_encoder,
    prompt_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, List[str]]]],
    cli_args: argparse.Namespace,
) -> Path:
    class_name = sample["cls_name"]
    image_path = sample["img_path"]
    display_image = load_display_image(image_path, args.img_size)

    normal_embeddings, abnormal_embeddings, prompt_texts = encode_prompt_groups(
        args,
        prompts,
        class_name,
        text_encoder,
        prompt_cache,
    )

    image_tensor = sample["img"].unsqueeze(0)
    with torch.inference_mode():
        _, patches = image_encoder(image_tensor)
        layer_maps = group_attention_maps(
            model,
            patches,
            normal_embeddings,
            abnormal_embeddings,
            aggregate=cli_args.aggregate,
        )

    image_stem = slugify(Path(image_path).stem)
    class_slug = slugify(class_name)
    prefix = f"{args.dataset_name}_{class_slug}_{sample_index:05d}_{image_stem}"
    output_dir = Path(cli_args.output_dir)
    figure_path = output_dir / f"{prefix}_prompt_patch_attention.png"
    maps_path = output_dir / f"{prefix}_prompt_patch_attention_maps.npz"
    metadata_path = output_dir / f"{prefix}_prompt_patch_attention.json"

    save_attention_figure(
        figure_path,
        display_image,
        layer_maps,
        args.vision_layers,
        topk=cli_args.topk,
        alpha=cli_args.alpha,
        normalization=cli_args.heatmap_normalization,
        signed_normalization=cli_args.signed_heatmap_normalization,
        signed_score=cli_args.signed_score,
        view=cli_args.view,
    )
    if cli_args.save_maps:
        save_numpy_maps(maps_path, layer_maps, cli_args.signed_score)

    metadata = {
        "dataset_name": args.dataset_name,
        "class_name": class_name,
        "sample_index": sample_index,
        "image_path": image_path,
        "checkpoint": cli_args.checkpoint,
        "img_size": args.img_size,
        "vision_layers": args.vision_layers,
        "aggregate": cli_args.aggregate,
        "view": cli_args.view,
        "signed_score": cli_args.signed_score,
        "signed_heatmap_normalization": cli_args.signed_heatmap_normalization,
        "topk": cli_args.topk,
        "positive_group": "normal prompts passed through positive_text_adapter",
        "negative_group": "abnormal prompts passed through negative_text_adapter",
        "normal_prompt_count": len(prompt_texts["normal"]),
        "abnormal_prompt_count": len(prompt_texts["abnormal"]),
        "layers": layer_metadata(layer_maps, args.vision_layers, cli_args.topk, cli_args.signed_score),
    }
    if cli_args.save_maps:
        metadata["maps_path"] = str(maps_path)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay per-layer patch attention to positive/negative prompt groups."
    )
    parser.add_argument("--dataset_name", default="mvtec", help="Dataset name used for class prompts.")
    parser.add_argument("--checkpoint", default="checkpoints_mvtec/model_epoch_10.pth")
    parser.add_argument("--checkpoint-args", default=None, help="Optional args.json path for model settings.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="prompt_patch_attention_vis")
    parser.add_argument("--base-dir", default="./")
    parser.add_argument("--model-id", default=DEFAULTS["model_id"])
    parser.add_argument("--vision-model-id", default=DEFAULTS["vision_model_id"])
    parser.add_argument("--vision-layers", type=parse_vision_layers, default=DEFAULTS["vision_layers"])
    parser.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    parser.add_argument("--out-dim", type=int, default=DEFAULTS["out_dim"])
    parser.add_argument("--global-topk-ratio", type=float, default=DEFAULTS["global_topk_ratio"])

    parser.add_argument("--image-path", default=None, help="Optional standalone image path.")
    parser.add_argument("--class-name", default=None, help="Class prompt to use or dataset class filter.")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset index to consider.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--anomaly",
        choices=["any", "normal", "abnormal"],
        default="any",
        help="Dataset sample filter.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["sum", "mean"],
        default="mean",
        help="How to aggregate attention over prompts in each group. Mean avoids prompt-count bias.",
    )
    parser.add_argument(
        "--view",
        choices=["signed", "split", "both"],
        default="signed",
        help="signed gives one red/green comparison map; split keeps separate positive and negative maps.",
    )
    parser.add_argument(
        "--signed-score",
        choices=["difference", "normalized-difference"],
        default="difference",
        help="Score used by signed view: positive-negative, or normalized by positive+negative.",
    )
    parser.add_argument("--topk", type=int, default=24, help="Number of highest-attention patches to outline.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay opacity.")
    parser.add_argument(
        "--heatmap-normalization",
        choices=["per-map", "none"],
        default="per-map",
        help="Normalization for split positive/negative heatmaps.",
    )
    parser.add_argument(
        "--signed-heatmap-normalization",
        choices=["symmetric", "none"],
        default="symmetric",
        help="Normalization for signed maps. Symmetric centers zero at white.",
    )
    parser.add_argument("--save-maps", action="store_true", help="Also save raw per-layer maps as .npz.")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    cli_args.device = maybe_use_cpu(cli_args.device)
    load_project_modules()

    checkpoint = Path(cli_args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    args = build_model_args(cli_args)
    args.device = cli_args.device

    dataset_constants = DatasetConstants(args.base_dir, args.dataset_name)
    prompts = prompt_generator(dataset_constants)
    prompt_cache = {}
    model = ZSADModel(args).to(args.device)
    state = load_checkpoint(checkpoint, args.device)
    model.load_state_dict(state)
    model.eval()

    image_encoder = DINOImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)
    text_encoder = CLIPTextEncoder(args.model_id, args.device)

    if cli_args.image_path:
        samples = [direct_image_sample(cli_args, args)]
    else:
        samples = list(dataset_samples(cli_args, args))
        if not samples:
            raise RuntimeError(
                "No dataset samples matched the requested filters. "
                "Try lowering --start-index, changing --class-name, or using --anomaly any."
            )

    saved = []
    for sample_index, sample in samples:
        figure_path = process_sample(
            sample_index,
            sample,
            args,
            model,
            image_encoder,
            prompts,
            text_encoder,
            prompt_cache,
            cli_args,
        )
        saved.append(figure_path)
        print(f"Saved {figure_path}")

    print(f"Done. Wrote {len(saved)} visualization(s) to {Path(cli_args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
