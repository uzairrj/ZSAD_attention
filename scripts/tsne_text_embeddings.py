from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

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


def _resolve_class_name(dataset_name: str, class_name: str | None, base_path: str) -> str:
    constants = DatasetConstants(base_path=base_path, dataset_name=dataset_name)
    class_names = constants.get_class_names()
    if class_name:
        lookup = {name.lower(): name for name in class_names}
        key = class_name.lower()
        if key not in lookup:
            raise ValueError(f"Class '{class_name}' not found for dataset '{dataset_name}'.")
        return lookup[key]
    return _prompt_choice("Select class:", class_names)


def _resolve_state(state: str | None) -> str:
    if state:
        state = state.lower()
        if state not in {"normal", "abnormal"}:
            raise ValueError("State must be 'normal' or 'abnormal'.")
        return state
    return _prompt_choice("Select state:", ["normal", "abnormal"])


def _validate_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested ({device}), but CUDA is not available.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute t-SNE for CLIP text embeddings of prompts."
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--state", type=str, choices=["normal", "abnormal"], default=None)
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--base-path", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "outputs" / "tsne_text_embeddings"))
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
    class_name = _resolve_class_name(dataset_name, args.class_name, args.base_path)
    state = _resolve_state(args.state)

    constants = DatasetConstants(base_path=args.base_path, dataset_name=dataset_name)
    per_template = args.per_template if args.per_template > 0 else None
    prompts = prompt_generator(constants, per_template=per_template, seed=args.seed)
    if state not in prompts or class_name not in prompts[state]:
        raise ValueError(f"No prompts found for {dataset_name}/{class_name} ({state}).")
    selected_prompts = prompts[state][class_name]

    encoder = CLIPTextEncoder(args.model_id, args.device)
    embeddings = encoder(selected_prompts).detach().cpu().numpy()
    prompts_for_tsne = list(selected_prompts)
    if args.include_average:
        average_embedding = embeddings.mean(axis=0, keepdims=True)
        embeddings = np.vstack([embeddings, average_embedding])
        prompts_for_tsne.append("[AVERAGE]")
    num_samples = embeddings.shape[0]
    if num_samples < 3:
        raise ValueError("t-SNE requires at least 3 prompt embeddings.")

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

    run_name = f"{dataset_name}_{class_name}_{state}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "tsne.npy", coords)

    with (output_dir / "tsne.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["prompt", "x", "y"])
        for prompt_text, (x, y) in zip(prompts_for_tsne, coords):
            writer.writerow([prompt_text, float(x), float(y)])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:-1, 0], coords[:-1, 1], s=150, alpha=0.5, edgecolors="blue")
    if args.include_average:
        avg_x, avg_y = coords[-1]
        ax.scatter(
            avg_x,
            avg_y,
            marker="*",
            s=220,
            color="red",
            edgecolors="black",
            linewidths=0.6,
            label="average",
        )
        ax.legend(frameon=False)
    ax.set_title(f"t-SNE of {state} prompts: {dataset_name}/{class_name}", fontsize=18)
    ax.set_ylabel("Component 2", fontsize=22)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_dir / "tsne.png", dpi=300)
    plt.close(fig)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
