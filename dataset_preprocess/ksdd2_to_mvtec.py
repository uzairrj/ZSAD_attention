#!/usr/bin/env python3
"""Convert KSDD2 into the MVTec AD directory layout.

The KSDD2 source folder is flat per split:

    train/10000.png
    train/10000_GT.png
    test/20000.png
    test/20000_GT.png

MVTec AD expects a class folder with ``train/good``, ``test/good``, and
``ground_truth/<defect_type>``. This converter treats masks with any non-zero
pixel as defects. Source train defects are moved to ``test/defect`` because
MVTec-style training contains only good samples.

Default behavior writes this converted class folder back into the KSDD2 root:

    KSDD2/
        train/
        test/
        SDD2/
            train/good/
            test/good/
            test/defect/
            ground_truth/defect/
        meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_KSDD2_ROOT = Path("/media/data/ukhan/data/computer_vision/KSDD2")
DEFAULT_CLASS_NAME = "SDD2"
DEFECT_NAME = "defect"
SOURCE_SPLITS = ("train", "test")
IMAGE_RE = re.compile(r"^\d+$")
MASK_RE = re.compile(r"^(\d+)_GT$")


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    source_split: str
    source_id: str
    is_defect: bool


@dataclass(frozen=True)
class DatasetPlan:
    train_good: list[Sample]
    test_good: list[Sample]
    test_defect: list[Sample]
    ignored: dict[str, list[Path]]


def canonical_images(split_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png" and IMAGE_RE.match(path.stem)
        ],
        key=lambda path: int(path.stem),
    )


def canonical_masks(split_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png" and MASK_RE.match(path.stem)
        ],
        key=lambda path: int(path.stem.removesuffix("_GT")),
    )


def ignored_pngs(split_dir: Path, images: list[Path], masks: list[Path]) -> list[Path]:
    used = set(images) | set(masks)
    return sorted(
        [
            path
            for path in split_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png" and path not in used
        ],
        key=lambda path: path.name,
    )


def mask_has_defect(mask_path: Path) -> bool:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return bool(mask.max() > 0)


def read_split_samples(src_root: Path, split_name: str) -> tuple[list[Sample], list[Path]]:
    split_dir = src_root / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing KSDD2 split directory: {split_dir}")

    images = canonical_images(split_dir)
    masks = canonical_masks(split_dir)
    mask_by_stem = {mask.stem.removesuffix("_GT"): mask for mask in masks}
    image_stems = {image.stem for image in images}

    missing_masks = [image for image in images if image.stem not in mask_by_stem]
    orphan_masks = [mask for stem, mask in mask_by_stem.items() if stem not in image_stems]
    if missing_masks:
        missing = ", ".join(path.name for path in missing_masks[:10])
        raise ValueError(f"{split_name}: images missing masks: {missing}")
    if orphan_masks:
        orphan = ", ".join(path.name for path in orphan_masks[:10])
        raise ValueError(f"{split_name}: masks missing images: {orphan}")

    samples = [
        Sample(
            image=image,
            mask=mask_by_stem[image.stem],
            source_split=split_name,
            source_id=image.stem,
            is_defect=mask_has_defect(mask_by_stem[image.stem]),
        )
        for image in images
    ]
    return samples, ignored_pngs(split_dir, images, masks)


def build_plan(src_root: Path) -> DatasetPlan:
    train_good: list[Sample] = []
    test_good: list[Sample] = []
    test_defect: list[Sample] = []
    ignored: dict[str, list[Path]] = {}

    for split_name in SOURCE_SPLITS:
        samples, ignored_files = read_split_samples(src_root, split_name)
        ignored[split_name] = ignored_files
        for sample in samples:
            if sample.is_defect:
                test_defect.append(sample)
            elif split_name == "train":
                train_good.append(sample)
            else:
                test_good.append(sample)

    return DatasetPlan(
        train_good=train_good,
        test_good=test_good,
        test_defect=test_defect,
        ignored=ignored,
    )


def output_exists(dst_root: Path, class_name: str) -> bool:
    class_dir = dst_root / class_name
    if (dst_root / "meta.json").exists():
        return True
    return any((class_dir / folder).exists() for folder in ("train", "test", "ground_truth"))


def clean_output(dst_root: Path, class_name: str) -> None:
    class_dir = dst_root / class_name
    for folder in ("train", "test", "ground_truth"):
        path = class_dir / folder
        if path.exists():
            shutil.rmtree(path)
    meta_path = dst_root / "meta.json"
    if meta_path.exists():
        meta_path.unlink()


def copy_sample(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def meta_record(
    class_name: str,
    rel_img: Path,
    rel_mask: Path | None,
    specie_name: str,
) -> dict[str, object]:
    return {
        "img_path": rel_img.as_posix(),
        "mask_path": rel_mask.as_posix() if rel_mask is not None else "",
        "cls_name": class_name,
        "specie_name": specie_name,
        "anomaly": 1 if rel_mask is not None else 0,
    }


def write_samples(
    dst_root: Path,
    class_name: str,
    plan: DatasetPlan,
    dry_run: bool,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    meta: dict[str, dict[str, list[dict[str, object]]]] = {
        "train": {class_name: []},
        "test": {class_name: []},
    }

    for index, sample in enumerate(plan.train_good):
        name = f"{index:04d}.png"
        rel_img = Path(class_name) / "train" / "good" / name
        copy_sample(sample.image, dst_root / rel_img, dry_run)
        meta["train"][class_name].append(meta_record(class_name, rel_img, None, "good"))

    for index, sample in enumerate(plan.test_good):
        name = f"{index:04d}.png"
        rel_img = Path(class_name) / "test" / "good" / name
        copy_sample(sample.image, dst_root / rel_img, dry_run)
        meta["test"][class_name].append(meta_record(class_name, rel_img, None, "good"))

    for index, sample in enumerate(plan.test_defect):
        name = f"{index:04d}.png"
        mask_name = f"{index:04d}_mask.png"
        rel_img = Path(class_name) / "test" / DEFECT_NAME / name
        rel_mask = Path(class_name) / "ground_truth" / DEFECT_NAME / mask_name
        copy_sample(sample.image, dst_root / rel_img, dry_run)
        copy_sample(sample.mask, dst_root / rel_mask, dry_run)
        meta["test"][class_name].append(meta_record(class_name, rel_img, rel_mask, DEFECT_NAME))

    return meta


def convert_ksdd2_to_mvtec(
    src_root: Path,
    dst_root: Path,
    class_name: str = DEFAULT_CLASS_NAME,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()

    plan = build_plan(src_root)
    if output_exists(dst_root, class_name):
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"MVTec output already exists under {dst_root / class_name}. "
                "Use --overwrite to replace generated train/test/ground_truth folders and meta.json."
            )
        if overwrite and not dry_run:
            clean_output(dst_root, class_name)

    meta = write_samples(dst_root, class_name, plan, dry_run)
    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        with (dst_root / "meta.json").open("w", encoding="utf-8") as file:
            file.write(json.dumps(meta, indent=4) + "\n")

    for split_name, files in plan.ignored.items():
        if files:
            names = ", ".join(path.name for path in files[:5])
            print(f"Ignored {len(files)} non-canonical {split_name} PNG file(s): {names}")

    print(
        f"{class_name}: train/good={len(plan.train_good)} "
        f"test/good={len(plan.test_good)} test/{DEFECT_NAME}={len(plan.test_defect)}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KSDD2 into the MVTec AD folder format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_KSDD2_ROOT,
        help=f"Original KSDD2 root. Default: {DEFAULT_KSDD2_ROOT}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Converted output root. Default: same as --src, adding an SDD2 class folder in-place.",
    )
    parser.add_argument(
        "--class-name",
        default=DEFAULT_CLASS_NAME,
        help=f"Single MVTec class folder name. Default: {DEFAULT_CLASS_NAME}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated train/test/ground_truth folders and meta.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pairings and print counts without copying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dst = args.dst if args.dst is not None else args.src
    convert_ksdd2_to_mvtec(
        src_root=args.src,
        dst_root=dst,
        class_name=args.class_name,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Converted"
    print(f"{action} KSDD2 MVTec layout at {dst}")


if __name__ == "__main__":
    main()
