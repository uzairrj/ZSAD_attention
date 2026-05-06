#!/usr/bin/env python3
"""Validate DTD-Synthetic and generate MVTec-style metadata.

DTD-Synthetic is already arranged like MVTec AD:

    Blotchy_099/
        train/good/
        test/good/
        test/bad/
        ground_truth/bad/
    meta.json

This script validates exact image/mask pairing, writes ``meta.json``, and can
optionally copy the same MVTec-style tree to another output root with ``--dst``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_DTD_ROOT = Path("/media/data/ukhan/data/computer_vision/DTD-Synthetic")
GOOD_NAME = "good"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: path.name,
    )


def discover_classes(src_root: Path) -> list[str]:
    class_names = sorted(
        [
            path.name
            for path in src_root.iterdir()
            if path.is_dir()
            and (path / "train").is_dir()
            and (path / "test").is_dir()
            and (path / "ground_truth").is_dir()
        ]
    )
    if not class_names:
        raise ValueError(f"No DTD-Synthetic class folders found in {src_root}")
    return class_names


def output_exists(dst_root: Path, class_names: list[str]) -> bool:
    if (dst_root / "meta.json").exists():
        return True
    return any(
        (dst_root / class_name / folder).exists()
        for class_name in class_names
        for folder in ("train", "test", "ground_truth")
    )


def clean_output(dst_root: Path, class_names: list[str]) -> None:
    for class_name in class_names:
        class_dir = dst_root / class_name
        for folder in ("train", "test", "ground_truth"):
            path = class_dir / folder
            if path.exists():
                shutil.rmtree(path)
    meta_path = dst_root / "meta.json"
    if meta_path.exists():
        meta_path.unlink()


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def meta_record(
    class_name: str,
    phase: str,
    specie_name: str,
    image_name: str,
    mask_name: str | None,
) -> dict[str, object]:
    rel_img = Path(class_name) / phase / specie_name / image_name
    rel_mask = (
        Path(class_name) / "ground_truth" / specie_name / mask_name
        if mask_name is not None
        else None
    )
    return {
        "img_path": rel_img.as_posix(),
        "mask_path": rel_mask.as_posix() if rel_mask is not None else "",
        "cls_name": class_name,
        "specie_name": specie_name,
        "anomaly": 0 if specie_name == GOOD_NAME else 1,
    }


def validate_and_copy_class(
    src_root: Path,
    dst_root: Path,
    class_name: str,
    copy_tree: bool,
    dry_run: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    train_info: list[dict[str, object]] = []
    test_info: list[dict[str, object]] = []
    counts = {"train_good": 0, "test_good": 0, "test_bad": 0}
    class_dir = src_root / class_name

    for image in iter_images(class_dir / "train" / GOOD_NAME):
        if copy_tree:
            rel_img = Path(class_name) / "train" / GOOD_NAME / image.name
            copy_file(image, dst_root / rel_img, dry_run)
        train_info.append(meta_record(class_name, "train", GOOD_NAME, image.name, None))
        counts["train_good"] += 1

    for image in iter_images(class_dir / "test" / GOOD_NAME):
        if copy_tree:
            rel_img = Path(class_name) / "test" / GOOD_NAME / image.name
            copy_file(image, dst_root / rel_img, dry_run)
        test_info.append(meta_record(class_name, "test", GOOD_NAME, image.name, None))
        counts["test_good"] += 1

    defect_dirs = sorted(
        [path for path in (class_dir / "test").iterdir() if path.is_dir() and path.name != GOOD_NAME],
        key=lambda path: path.name,
    )
    for defect_dir in defect_dirs:
        specie_name = defect_dir.name
        mask_dir = class_dir / "ground_truth" / specie_name
        for image in iter_images(defect_dir):
            mask_name = f"{image.stem}_mask.png"
            mask = mask_dir / mask_name
            if not mask.is_file():
                raise FileNotFoundError(f"Missing mask for {image}: {mask}")
            if copy_tree:
                rel_img = Path(class_name) / "test" / specie_name / image.name
                rel_mask = Path(class_name) / "ground_truth" / specie_name / mask.name
                copy_file(image, dst_root / rel_img, dry_run)
                copy_file(mask, dst_root / rel_mask, dry_run)
            test_info.append(meta_record(class_name, "test", specie_name, image.name, mask.name))
            counts["test_bad"] += 1

        expected_masks = {f"{image.stem}_mask.png" for image in iter_images(defect_dir)}
        actual_masks = {mask.name for mask in iter_images(mask_dir)}
        extra_masks = sorted(actual_masks - expected_masks)
        if extra_masks:
            examples = ", ".join(extra_masks[:10])
            raise ValueError(f"{class_name}/{specie_name}: masks without images: {examples}")

    return train_info, test_info, counts


def convert_dtd_synthetic_to_mvtec(
    src_root: Path,
    dst_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    class_names = discover_classes(src_root)
    copy_tree = src_root != dst_root

    if copy_tree and output_exists(dst_root, class_names):
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"MVTec output already exists under {dst_root}. "
                "Use --overwrite to replace generated train/test/ground_truth folders and meta.json."
            )
        if overwrite and not dry_run:
            clean_output(dst_root, class_names)

    meta: dict[str, dict[str, list[dict[str, object]]]] = {"train": {}, "test": {}}
    totals = {"train_good": 0, "test_good": 0, "test_bad": 0}
    for class_name in class_names:
        train_info, test_info, counts = validate_and_copy_class(
            src_root=src_root,
            dst_root=dst_root,
            class_name=class_name,
            copy_tree=copy_tree,
            dry_run=dry_run,
        )
        meta["train"][class_name] = train_info
        meta["test"][class_name] = test_info
        for key, value in counts.items():
            totals[key] += value
        print(
            f"{class_name}: train/good={counts['train_good']} "
            f"test/good={counts['test_good']} test/bad={counts['test_bad']}"
        )

    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        with (dst_root / "meta.json").open("w", encoding="utf-8") as file:
            file.write(json.dumps(meta, indent=4) + "\n")

    print(
        f"Total: train/good={totals['train_good']} "
        f"test/good={totals['test_good']} test/bad={totals['test_bad']}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DTD-Synthetic and generate MVTec-style meta.json."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_DTD_ROOT,
        help=f"DTD-Synthetic root. Default: {DEFAULT_DTD_ROOT}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output root. Default: same as --src, generating meta.json in place.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When --dst differs from --src, replace generated output folders and meta.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate layout and print counts without copying files or writing meta.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dst = args.dst if args.dst is not None else args.src
    convert_dtd_synthetic_to_mvtec(
        src_root=args.src,
        dst_root=dst,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Prepared"
    print(f"{action} DTD-Synthetic MVTec layout at {dst}")


if __name__ == "__main__":
    main()
