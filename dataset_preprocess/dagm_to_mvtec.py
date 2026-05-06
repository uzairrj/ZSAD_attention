#!/usr/bin/env python3
"""Convert DAGM into the MVTec AD directory layout.

The source DAGM release keeps anomalous samples in both ``Train`` and ``Test``.
MVTec AD expects ``train/good`` to contain only normal images, while all
anomalous images live under ``test/<defect_type>`` with masks under
``ground_truth/<defect_type>``.

Default behavior writes the converted MVTec-style folders back into the DAGM
root, next to the original uppercase ``Train``/``Test`` source folders:

    Class1/
        Train/
        Test/
        train/good/
        test/good/
        test/defect/
        ground_truth/defect/
    meta.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DAGM_ROOT = Path("/media/data/ukhan/data/computer_vision/DAGM_anomaly_detection")
CLASS_NAMES = [f"Class{i}" for i in range(1, 11)]
SOURCE_SPLITS = ("Train", "Test")
DEFECT_NAME = "defect"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    source_split: str

    @property
    def is_defect(self) -> bool:
        return self.mask is not None


@dataclass(frozen=True)
class ClassPlan:
    train_good: list[Sample]
    test_good: list[Sample]
    test_defect: list[Sample]


def natural_class_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("Class")
    return (int(suffix), path.name) if suffix.isdigit() else (10_000, path.name)


def iter_images(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda path: path.name,
    )


def read_split_samples(class_dir: Path, split_name: str) -> tuple[list[Sample], list[Sample]]:
    split_dir = class_dir / split_name
    label_dir = split_dir / "Label"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing DAGM split directory: {split_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing DAGM label directory: {label_dir}")

    images = iter_images(split_dir)
    masks = iter_images(label_dir)
    image_by_stem = {image.stem: image for image in images}
    mask_by_stem: dict[str, Path] = {}

    for mask in masks:
        if not mask.stem.endswith("_label"):
            raise ValueError(f"Unexpected DAGM mask name, expected '*_label': {mask}")
        image_stem = mask.stem[: -len("_label")]
        if image_stem not in image_by_stem:
            raise ValueError(f"Mask has no matching source image: {mask}")
        mask_by_stem[image_stem] = mask

    good: list[Sample] = []
    defect: list[Sample] = []
    for image in images:
        sample = Sample(image=image, mask=mask_by_stem.get(image.stem), source_split=split_name)
        if sample.is_defect:
            defect.append(sample)
        else:
            good.append(sample)

    return good, defect


def build_class_plan(class_dir: Path) -> ClassPlan:
    train_good: list[Sample] = []
    test_good: list[Sample] = []
    test_defect: list[Sample] = []

    for split_name in SOURCE_SPLITS:
        good, defect = read_split_samples(class_dir, split_name)
        test_good.extend(good[: len(defect)])
        train_good.extend(good[len(defect) :])
        test_defect.extend(defect)

    return ClassPlan(
        train_good=train_good,
        test_good=test_good,
        test_defect=test_defect,
    )


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


def copy_sample(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_class(
    dst_root: Path,
    class_name: str,
    plan: ClassPlan,
    dry_run: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_info: list[dict[str, object]] = []
    test_info: list[dict[str, object]] = []

    for index, sample in enumerate(plan.train_good):
        name = f"{index:04d}.png"
        rel_img = Path(class_name) / "train" / "good" / name
        copy_sample(sample.image, dst_root / rel_img, dry_run)
        train_info.append(
            {
                "img_path": rel_img.as_posix(),
                "mask_path": "",
                "cls_name": class_name,
                "specie_name": "good",
                "anomaly": 0,
            }
        )

    for index, sample in enumerate(plan.test_good):
        name = f"{index:04d}.png"
        rel_img = Path(class_name) / "test" / "good" / name
        copy_sample(sample.image, dst_root / rel_img, dry_run)
        test_info.append(
            {
                "img_path": rel_img.as_posix(),
                "mask_path": "",
                "cls_name": class_name,
                "specie_name": "good",
                "anomaly": 0,
            }
        )

    for index, sample in enumerate(plan.test_defect):
        if sample.mask is None:
            raise ValueError(f"Defect sample is missing a mask: {sample.image}")

        name = f"{index:04d}.png"
        mask_name = f"{index:04d}_mask.png"
        rel_img = Path(class_name) / "test" / DEFECT_NAME / name
        rel_mask = Path(class_name) / "ground_truth" / DEFECT_NAME / mask_name
        copy_sample(sample.image, dst_root / rel_img, dry_run)
        copy_sample(sample.mask, dst_root / rel_mask, dry_run)
        test_info.append(
            {
                "img_path": rel_img.as_posix(),
                "mask_path": rel_mask.as_posix(),
                "cls_name": class_name,
                "specie_name": DEFECT_NAME,
                "anomaly": 1,
            }
        )

    return train_info, test_info


def discover_classes(src_root: Path) -> list[str]:
    found = sorted(
        [path.name for path in src_root.iterdir() if path.is_dir() and path.name.startswith("Class")],
        key=lambda name: natural_class_key(Path(name)),
    )
    missing = [class_name for class_name in CLASS_NAMES if class_name not in found]
    if missing:
        raise FileNotFoundError(f"Missing DAGM class directories: {', '.join(missing)}")
    return CLASS_NAMES


def convert_dagm_to_mvtec(
    src_root: Path,
    dst_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    class_names = discover_classes(src_root)

    if output_exists(dst_root, class_names):
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"MVTec output already exists under {dst_root}. "
                "Use --overwrite to replace only generated train/test/ground_truth folders and meta.json."
            )
        if overwrite and not dry_run:
            clean_output(dst_root, class_names)

    meta: dict[str, dict[str, list[dict[str, object]]]] = {"train": {}, "test": {}}
    for class_name in class_names:
        plan = build_class_plan(src_root / class_name)
        train_info, test_info = write_class(dst_root, class_name, plan, dry_run)
        meta["train"][class_name] = train_info
        meta["test"][class_name] = test_info
        print(
            f"{class_name}: train/good={len(plan.train_good)} "
            f"test/good={len(plan.test_good)} test/{DEFECT_NAME}={len(plan.test_defect)}"
        )

    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        with (dst_root / "meta.json").open("w", encoding="utf-8") as file:
            file.write(json.dumps(meta, indent=4) + "\n")

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DAGM_anomaly_detection into the MVTec AD folder format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_DAGM_ROOT,
        help=f"Original DAGM root. Default: {DEFAULT_DAGM_ROOT}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Converted output root. Default: same as --src, adding MVTec folders in-place.",
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
    convert_dagm_to_mvtec(
        src_root=args.src,
        dst_root=dst,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Converted"
    print(f"{action} DAGM MVTec layout at {dst}")


if __name__ == "__main__":
    main()
