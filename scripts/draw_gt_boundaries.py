from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gt_boundaries"
RED_RGB = np.array([255, 0, 0], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw red ground-truth mask boundaries on images listed in meta.json."
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        required=True,
        help="Path to the dataset meta.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for boundary visualizations. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Metadata split to visualize, or 'all'. Default: test.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional class names to visualize. Default: all classes.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Boundary line thickness in pixels. Default: 2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to write.",
    )
    parser.add_argument(
        "--include-normal",
        action="store_true",
        help="Also save samples without masks. They are written unchanged.",
    )
    return parser.parse_args()


def load_meta(meta_json: Path) -> dict:
    with meta_json.open("r", encoding="utf-8") as file:
        return json.load(file)


def selected_splits(meta: dict, split: str) -> list[str]:
    if split == "all":
        return list(meta.keys())
    if split not in meta:
        available = ", ".join(meta.keys())
        raise KeyError(f"Split '{split}' not found in meta.json. Available: {available}")
    return [split]


def iter_records(meta: dict, splits: list[str], classes: set[str] | None):
    for split in splits:
        split_data = meta[split]
        if not isinstance(split_data, dict):
            raise TypeError(f"Expected split '{split}' to contain a class dictionary.")

        for class_name, records in split_data.items():
            if classes is not None and class_name not in classes:
                continue
            for record in records:
                yield split, class_name, record


def resolve_dataset_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def read_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def read_mask(mask_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        if mask.size != image_size:
            mask = mask.resize(image_size, Image.Resampling.NEAREST)
        return np.asarray(mask) > 0


def boundary_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    height, width = mask.shape

    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[y_offset : y_offset + height, x_offset : x_offset + width]

    boundary = mask & ~eroded
    if thickness <= 1 or not boundary.any():
        return boundary

    radius = max(thickness // 2, 1)
    boundary_image = Image.fromarray(boundary.astype(np.uint8) * 255, mode="L")
    boundary_image = boundary_image.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    return np.asarray(boundary_image) > 0


def draw_red_boundary(image: Image.Image, mask: np.ndarray, thickness: int) -> Image.Image:
    output = np.asarray(image).copy()
    output[boundary_mask(mask, thickness)] = RED_RGB
    return Image.fromarray(output, mode="RGB")


def output_path(output_dir: Path, split: str, class_name: str, record: dict) -> Path:
    image_rel = Path(record["img_path"])
    specie_name = record.get("specie_name") or image_rel.parent.name
    filename = f"{image_rel.stem}_gt_boundary.png"
    return output_dir / split / class_name / specie_name / filename


def main() -> None:
    args = parse_args()
    meta_json = args.meta_json.resolve()
    dataset_root = meta_json.parent
    output_dir = args.output_dir.resolve()
    class_filter = set(args.classes) if args.classes else None

    meta = load_meta(meta_json)
    splits = selected_splits(meta, args.split)

    written = 0
    skipped_no_mask = 0
    skipped_errors = 0

    for split, class_name, record in iter_records(meta, splits, class_filter):
        mask_rel = record.get("mask_path", "")
        if not mask_rel and not args.include_normal:
            skipped_no_mask += 1
            continue

        try:
            image_path = resolve_dataset_path(dataset_root, record["img_path"])
            image = read_image(image_path)

            if mask_rel:
                mask_path = resolve_dataset_path(dataset_root, mask_rel)
                mask = read_mask(mask_path, image.size)
                result = draw_red_boundary(image, mask, max(args.thickness, 1))
            else:
                result = image

            save_path = output_path(output_dir, split, class_name, record)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(save_path)
            written += 1

            if args.limit is not None and written >= args.limit:
                break
        except (KeyError, OSError, ValueError) as exc:
            skipped_errors += 1
            print(f"warning: skipped record because {exc}", file=sys.stderr)

    print(f"Wrote {written} image(s) to {output_dir}")
    if skipped_no_mask:
        print(f"Skipped {skipped_no_mask} sample(s) without masks.")
    if skipped_errors:
        print(f"Skipped {skipped_errors} sample(s) because of read/write errors.")


if __name__ == "__main__":
    main()
