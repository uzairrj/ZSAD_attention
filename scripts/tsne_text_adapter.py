from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "scikit-learn is required for t-SNE. Install it with `pip install scikit-learn`."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbones.CLIP import CLIPTextEncoder
from datasets.constants import DatasetConstants
from model.model import ZSADModel
from utils.args import Args
from utils.utils import prompt_generator


def _validate_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested ({device}), but CUDA is not available.")


def _load_model(checkpoint_path: Path, args: Args, device: str) -> ZSADModel:
    """Load a ZSAD model from checkpoint."""
    model = ZSADModel(args)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model.to(device)
    model.eval()
    return model


def _plot_tsne_all_classes(
    coords: np.ndarray,
    labels: list[tuple[str, str]],
    title: str,
    output_path: Path,
) -> None:
    """Plot t-SNE with all classes and normal/abnormal in different colors/markers."""
    unique_classes = sorted(set(c for c, _ in labels))
    unique_states = sorted(set(s for _, s in labels))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    marker_map = {"normal": "o", "abnormal": "s"}
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for class_name in unique_classes:
        for state in unique_states:
            mask = np.array([(c == class_name and s == state) for c, s in labels])
            if mask.sum() == 0:
                continue
            
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=100,
                alpha=0.6,
                color=color_map[class_name],
                marker=marker_map[state],
                label=f"{class_name} ({state})",
                edgecolors="black",
                linewidths=0.5,
            )
    
    ax.set_xlabel("Component 1", fontsize=22)
    ax.set_ylabel("Component 2", fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_title(title, fontsize=18)
    ax.legend(frameon=False, fontsize=10, loc="best", ncol=2)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute t-SNE of text embeddings (normal+abnormal) after adapter for all classes."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mvtec",
        help="Dataset name (default: mvtec).",
    )
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--base-path", type=str, default=".")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(ROOT / "checkpoints_visa"),
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model_epoch_5.pth",
        help="Checkpoint filename.",
    )
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "tsne_adapter"))
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument(
        "--per-template",
        type=int,
        default=0,
        help="Sample this many prompts per template (0 = use all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _validate_device(args.device)

    dataset_name = args.dataset_name.lower()
    constants = DatasetConstants(base_path=args.base_path, dataset_name=dataset_name)
    class_names = constants.get_class_names()

    per_template = args.per_template if args.per_template > 0 else None
    prompts = prompt_generator(constants, per_template=per_template, seed=args.seed)

    encoder = CLIPTextEncoder(args.model_id, args.device)

    model_args = Args(
        out_dim=768,
        vision_layers=[6, 12, 18, 24],
        img_size=768,
        global_topk_ratio=0.01,
    )
    checkpoint_path = Path(args.checkpoint_dir) / args.checkpoint_name
    model = _load_model(checkpoint_path, model_args, args.device)

    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings_list = []
    all_labels_list = []

    pbar = tqdm(total=len(class_names), desc="Collecting embeddings (post-adapter)")

    for class_name in class_names:
        normal_prompts = prompts["normal"][class_name]
        abnormal_prompts = prompts["abnormal"][class_name]

        normal_embeddings = encoder(normal_prompts).detach()
        abnormal_embeddings = encoder(abnormal_prompts).detach()

        with torch.inference_mode():
            normal_adapted = model.positive_text_adapter(normal_embeddings.to(args.device)).cpu().numpy()
            abnormal_adapted = model.negative_text_adapter(abnormal_embeddings.to(args.device)).cpu().numpy()

        all_embeddings_list.append(normal_adapted)
        all_embeddings_list.append(abnormal_adapted)

        all_labels_list.extend([(class_name, "normal")] * len(normal_adapted))
        all_labels_list.extend([(class_name, "abnormal")] * len(abnormal_adapted))

        pbar.update(1)

    pbar.close()

    all_embeddings = np.vstack(all_embeddings_list)
    num_samples = all_embeddings.shape[0]

    print(f"Total samples: {num_samples}")

    perplexity = min(args.perplexity, num_samples - 1)
    if perplexity < 2:
        perplexity = 2

    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        init="pca",
        metric="cosine",
    )
    coords = tsne.fit_transform(all_embeddings)

    title = f"All Classes - Normal vs Abnormal (Post-Adapter) - {dataset_name}"
    img_name = f"all_classes.png"
    _plot_tsne_all_classes(
        coords,
        all_labels_list,
        title,
        output_dir / img_name,
    )

    print(f"Saved t-SNE plot to: {output_dir / img_name}")


if __name__ == "__main__":
    main()
