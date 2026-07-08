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
from utils.utils import prompt_generator


def _prompt_choice(prompt_text: str, options: list[str]) -> str:
    print(prompt_text)
    for idx, opt in enumerate(options, 1):
        print(f"{idx}. {opt}")
    option_map = {opt.lower(): opt for opt in options}
    while True:
        selection = input("Enter number or name: ").strip()
        if selection.isdigit():
            index = int(selection) - 1
            if 0 <= index < len(options):
                return options[index]
        else:
            key = selection.lower()
            if key in option_map:
                return option_map[key]
        print("Invalid selection. Try again.")


def _resolve_dataset_name(dataset_name: str | None, base_path: str) -> str:
    temp_constants = DatasetConstants(base_path=base_path, dataset_name="mvtec")
    available = sorted(temp_constants.CLASS_NAMES.keys())
    if dataset_name:
        dataset_key = dataset_name.lower()
        if dataset_key not in temp_constants.CLASS_NAMES:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {', '.join(available)}"
            )
        return dataset_key
    selection = _prompt_choice("Select dataset:", available)
    return selection.lower()


def _validate_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested ({device}), but CUDA is not available.")


def _plot_tsne(
    coords: np.ndarray,
    title: str,
    output_path: Path,
    include_average: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_box_aspect(1)

    # 1. Force the axis grids and ticks to the background layer
    ax.set_axisbelow(True)

    # 2. Assign a higher zorder (e.g., 3 and 4) so circles sit above the grid (zorder=2.5)
    ax.scatter(
        coords[:-1, 0] if include_average else coords[:, 0],
        coords[:-1, 1] if include_average else coords[:, 1],
        s=550,
        alpha=0.9,
        c="#2888dc",
        edgecolors="#0264ba",
        linewidths=1.5,
        zorder=3,  # Sits above the grid
    )

    if include_average:
        avg_x, avg_y = coords[-1]
        ax.scatter(
            avg_x,
            avg_y,
            marker="*",
            s=850,
            color="#FAD7AC",
            edgecolors="#B46504",
            linewidths=1.0,
            label="average",
            zorder=4,  # Sits above the grid and standard circles
        )
        
    ax.set_frame_on(False)
    ax.minorticks_on()

    ax.tick_params(
        axis="both",          
        which="both",         
        bottom=False,         
        left=False,          
        labelbottom=False,    
        labelleft=False       
    )

    ax.grid(visible=True, which='major', color='#b0c4de', linestyle='-', linewidth=0.8, alpha=0.7)
    ax.grid(visible=True, which='minor', color='#b0c4de', linestyle='-', linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute t-SNE for CLIP text embeddings of all classes and states in a dataset."
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--base-path", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "tsne_all"))
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument(
        "--per-template",
        type=int,
        default=0,
        help="Sample this many prompts per template (0 = use all).",
    )
    parser.add_argument(
        "--include-average",
        action="store_true",
        help="Include the average embedding as a star in the t-SNE plot.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _validate_device(args.device)

    dataset_name = _resolve_dataset_name(args.dataset_name, args.base_path)

    constants = DatasetConstants(base_path=args.base_path, dataset_name=dataset_name)
    class_names = constants.get_class_names()

    per_template = args.per_template if args.per_template > 0 else None
    prompts = prompt_generator(constants, per_template=per_template, seed=args.seed)

    encoder = CLIPTextEncoder(args.model_id, args.device)

    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    total_plots = len(class_names) * 2  # 2 states per class
    pbar = tqdm(total=total_plots, desc="Generating t-SNE plots")

    for class_name in class_names:
        for state in ["normal", "abnormal"]:
            if state not in prompts or class_name not in prompts[state]:
                pbar.update(1)
                continue

            selected_prompts = prompts[state][class_name]
            embeddings = encoder(selected_prompts).detach().cpu().numpy()
            prompts_for_plot = list(selected_prompts)

            if args.include_average:
                average_embedding = embeddings.mean(axis=0, keepdims=True)
                embeddings = np.vstack([embeddings, average_embedding])
                prompts_for_plot.append("[AVERAGE]")

            num_samples = embeddings.shape[0]
            if num_samples < 3:
                print(f"Warning: skipping {class_name}/{state} (< 3 samples)")
                pbar.update(1)
                continue

            perplexity = min(args.perplexity, num_samples - 1)
            if perplexity < 2:
                perplexity = 2

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=args.seed,
                init="pca",
                metric="cosine",
            )
            coords = tsne.fit_transform(embeddings)

            title = f"{class_name} - {state}"
            img_name = f"{class_name}_{state}.png"
            _plot_tsne(
                coords,
                title,
                output_dir / img_name,
                include_average=args.include_average,
            )

            pbar.update(1)

    pbar.close()
    print(f"Saved all t-SNE plots to: {output_dir}")


if __name__ == "__main__":
    main()
