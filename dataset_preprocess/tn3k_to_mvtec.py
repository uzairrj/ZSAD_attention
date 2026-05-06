#!/usr/bin/env python3
"""Convert TN3K into the MVTec AD directory layout.

Expected TN3K source layout:

    tn3k/
        trainval-image/
        trainval-mask/
        test-image/
        test-mask/
        tn3k-trainval-fold0.json

The current medical datasets in this repository use MVTec-style metadata with
segmentation samples under ``test/defect`` and masks under
``ground_truth/defect``. By default this script converts the official TN3K test
split into:

    TN3K/
        tn3k/
            train/
            test/defect/
            ground_truth/defect/
        meta.json

Fold-based train/validation conversion is available through ``--train-source``
and ``--test-source``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


DEFAULT_TN3K_ROOT = Path("/media/data/ukhan/data/medical_cv/tn3k")
DEFAULT_MVTEC_ROOT = Path("/media/data/ukhan/data/medical_cv/TN3K")
DEFAULT_CLASS_NAME = "tn3k"
DEFAULT_DEFECT_NAME = "defect"
DEFAULT_FOLD_JSON = "tn3k-trainval-fold0.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    source_name: str
    source_id: str
    image_suffix: str


def image_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def index_image_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in files:
            duplicates.setdefault(path.stem, [files[path.stem].name]).append(path.name)
            continue
        files[path.stem] = path

    if duplicates:
        stem, names = next(iter(duplicates.items()))
        raise ValueError(
            f"Multiple files with stem {stem!r} under {directory}: {', '.join(names)}"
        )
    return files


def indexed_file(files: dict[str, Path], directory: Path, stem: str) -> Path:
    try:
        return files[stem]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Missing file with stem {stem!r} under {directory}"
        ) from exc


def validate_source_dirs(src_root: Path) -> None:
    for folder in ("trainval-image", "trainval-mask", "test-image", "test-mask"):
        path = src_root / folder
        if not path.is_dir():
            raise FileNotFoundError(f"Missing TN3K source directory: {path}")


def validate_pair_dimensions(image: Path, mask: Path) -> None:
    with Image.open(image) as image_file, Image.open(mask) as mask_file:
        if image_file.size != mask_file.size:
            raise ValueError(
                f"Image/mask size mismatch for {image.name}: "
                f"{image_file.size} vs {mask_file.size}"
            )


def make_sample(
    image_by_stem: dict[str, Path],
    mask_by_stem: dict[str, Path],
    image_dir: Path,
    mask_dir: Path,
    stem: str,
    source_name: str,
    validate_size: bool,
) -> Sample:
    image = indexed_file(image_by_stem, image_dir, stem)
    mask = indexed_file(mask_by_stem, mask_dir, stem)
    if validate_size:
        validate_pair_dimensions(image, mask)
    return Sample(
        image=image,
        mask=mask,
        source_name=source_name,
        source_id=stem,
        image_suffix=image.suffix.lower(),
    )


def load_fold(fold_json: Path) -> dict[str, list[int]]:
    if not fold_json.is_file():
        raise FileNotFoundError(f"Missing TN3K fold JSON: {fold_json}")
    with fold_json.open("r", encoding="utf-8") as file:
        fold = json.load(file)

    for key in ("train", "val"):
        if key not in fold or not isinstance(fold[key], list):
            raise ValueError(f"{fold_json} must contain a list named {key!r}")
        if len(fold[key]) != len(set(fold[key])):
            raise ValueError(f"{fold_json}: duplicate indices found in {key!r}")

    overlap = set(fold["train"]) & set(fold["val"])
    if overlap:
        examples = ", ".join(str(index) for index in sorted(overlap)[:10])
        raise ValueError(f"{fold_json}: train/val overlap: {examples}")
    return fold


def samples_from_indices(
    src_root: Path,
    indices: list[int],
    source_name: str,
    validate_size: bool,
) -> list[Sample]:
    image_dir = src_root / "trainval-image"
    mask_dir = src_root / "trainval-mask"
    image_by_stem = index_image_files(image_dir)
    mask_by_stem = index_image_files(mask_dir)
    return [
        make_sample(
            image_by_stem,
            mask_by_stem,
            image_dir,
            mask_dir,
            f"{index:04d}",
            source_name,
            validate_size,
        )
        for index in indices
    ]


def samples_from_directory(
    src_root: Path,
    image_folder: str,
    mask_folder: str,
    source_name: str,
    validate_size: bool,
) -> list[Sample]:
    image_dir = src_root / image_folder
    mask_dir = src_root / mask_folder
    image_by_stem = index_image_files(image_dir)
    mask_by_stem = index_image_files(mask_dir)
    images = sorted(image_by_stem.values(), key=image_sort_key)
    if not images:
        raise ValueError(f"No images found under {image_dir}")

    samples = [
        make_sample(
            image_by_stem,
            mask_by_stem,
            image_dir,
            mask_dir,
            image.stem,
            source_name,
            validate_size,
        )
        for image in images
    ]
    image_stems = {sample.source_id for sample in samples}
    extra_masks = sorted(
        path.name for stem, path in mask_by_stem.items() if stem not in image_stems
    )
    if extra_masks:
        examples = ", ".join(extra_masks[:10])
        raise ValueError(f"Masks without images under {mask_dir}: {examples}")
    return samples


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


def copy_image(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_binary_mask(src: Path, dst: Path, threshold: int, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as mask_file:
        mask = mask_file.convert("L")
        mask = mask.point(lambda value: 255 if value > threshold else 0)
        mask.save(dst)


def meta_record(
    class_name: str,
    defect_name: str,
    rel_img: Path,
    rel_mask: Path,
) -> dict[str, object]:
    return {
        "img_path": rel_img.as_posix(),
        "mask_path": rel_mask.as_posix(),
        "cls_name": class_name,
        "specie_name": defect_name,
        "anomaly": 1,
    }


def write_split(
    dst_root: Path,
    class_name: str,
    defect_name: str,
    phase: str,
    samples: list[Sample],
    mask_threshold: int,
    dry_run: bool,
) -> list[dict[str, object]]:
    split_info: list[dict[str, object]] = []
    for sample in samples:
        output_stem = f"{phase}_{sample.source_name}_{sample.source_id}"
        image_name = f"{output_stem}{sample.image_suffix}"
        mask_name = f"{output_stem}_mask.png"
        rel_img = Path(class_name) / phase / defect_name / image_name
        rel_mask = Path(class_name) / "ground_truth" / defect_name / mask_name

        copy_image(sample.image, dst_root / rel_img, dry_run)
        write_binary_mask(sample.mask, dst_root / rel_mask, mask_threshold, dry_run)
        split_info.append(meta_record(class_name, defect_name, rel_img, rel_mask))

    return split_info


def build_train_samples(
    src_root: Path,
    fold: dict[str, list[int]] | None,
    train_source: str,
    validate_size: bool,
) -> list[Sample]:
    if train_source == "none":
        return []
    if train_source == "fold-train":
        if fold is None:
            raise ValueError("--train-source fold-train requires --fold-json")
        return samples_from_indices(src_root, fold["train"], "trainval", validate_size)
    if train_source == "trainval":
        return samples_from_directory(
            src_root, "trainval-image", "trainval-mask", "trainval", validate_size
        )
    raise ValueError(f"Unsupported train source: {train_source}")


def build_test_samples(
    src_root: Path,
    fold: dict[str, list[int]] | None,
    test_source: str,
    validate_size: bool,
) -> list[Sample]:
    official = lambda: samples_from_directory(
        src_root, "test-image", "test-mask", "official", validate_size
    )
    trainval = lambda: samples_from_directory(
        src_root, "trainval-image", "trainval-mask", "trainval", validate_size
    )

    if test_source == "official":
        return official()
    if test_source == "trainval":
        return trainval()
    if test_source == "all":
        return trainval() + official()
    if test_source in {"fold-val", "fold-val+official"}:
        if fold is None:
            raise ValueError(f"--test-source {test_source} requires --fold-json")
        samples = samples_from_indices(src_root, fold["val"], "trainval", validate_size)
        if test_source == "fold-val+official":
            samples.extend(official())
        return samples
    raise ValueError(f"Unsupported test source: {test_source}")


def convert_tn3k_to_mvtec(
    src_root: Path,
    dst_root: Path,
    fold_json: Path | None,
    class_name: str = DEFAULT_CLASS_NAME,
    defect_name: str = DEFAULT_DEFECT_NAME,
    train_source: str = "none",
    test_source: str = "official",
    mask_threshold: int = 127,
    validate_size: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    if fold_json is not None:
        fold_json = fold_json.expanduser()
        if not fold_json.is_absolute():
            fold_json = src_root / fold_json
        fold_json = fold_json.resolve()

    validate_source_dirs(src_root)
    needs_fold = train_source == "fold-train" or test_source.startswith("fold-val")
    fold = load_fold(fold_json) if needs_fold and fold_json is not None else None
    if needs_fold and fold is None:
        raise ValueError("A fold JSON is required for the selected train/test source")

    if output_exists(dst_root, class_name):
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"MVTec output already exists under {dst_root / class_name}. "
                "Use --overwrite to replace generated train/test/ground_truth folders and meta.json."
            )
        if overwrite and not dry_run:
            clean_output(dst_root, class_name)

    train_samples = build_train_samples(src_root, fold, train_source, validate_size)
    test_samples = build_test_samples(src_root, fold, test_source, validate_size)

    meta: dict[str, dict[str, list[dict[str, object]]]] = {
        "train": {class_name: []},
        "test": {class_name: []},
    }
    meta["train"][class_name] = write_split(
        dst_root, class_name, defect_name, "train", train_samples, mask_threshold, dry_run
    )
    meta["test"][class_name] = write_split(
        dst_root, class_name, defect_name, "test", test_samples, mask_threshold, dry_run
    )

    if not dry_run:
        (dst_root / class_name / "train").mkdir(parents=True, exist_ok=True)
        (dst_root / class_name / "test" / defect_name).mkdir(parents=True, exist_ok=True)
        (dst_root / class_name / "ground_truth" / defect_name).mkdir(
            parents=True, exist_ok=True
        )
        with (dst_root / "meta.json").open("w", encoding="utf-8") as file:
            file.write(json.dumps(meta, indent=4) + "\n")

    print(
        f"{class_name}: train/{defect_name}={len(train_samples)} "
        f"test/{defect_name}={len(test_samples)}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TN3K to an MVTec-style layout.")
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_TN3K_ROOT,
        help=f"Raw TN3K root. Default: {DEFAULT_TN3K_ROOT}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_MVTEC_ROOT,
        help=f"MVTec-style output root. Default: {DEFAULT_MVTEC_ROOT}",
    )
    parser.add_argument(
        "--fold-json",
        type=Path,
        default=Path(DEFAULT_FOLD_JSON),
        help=(
            "TN3K train/validation fold JSON, relative to --src or absolute. "
            f"Default: {DEFAULT_FOLD_JSON}"
        ),
    )
    parser.add_argument(
        "--class-name",
        default=DEFAULT_CLASS_NAME,
        help=f"Output MVTec class name. Default: {DEFAULT_CLASS_NAME}",
    )
    parser.add_argument(
        "--defect-name",
        default=DEFAULT_DEFECT_NAME,
        help=f"Output defect folder/specie name. Default: {DEFAULT_DEFECT_NAME}",
    )
    parser.add_argument(
        "--train-source",
        choices=("none", "fold-train", "trainval"),
        default="none",
        help="Samples to write under train/<defect>. Default: none.",
    )
    parser.add_argument(
        "--test-source",
        choices=("official", "fold-val", "fold-val+official", "trainval", "all"),
        default="official",
        help="Samples to write under test/<defect>. Default: official.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Threshold used when binarizing TN3K JPEG masks to PNG. Default: 127.",
    )
    parser.add_argument(
        "--skip-size-validation",
        action="store_true",
        help="Skip image/mask dimension checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated train/test/ground_truth folders and meta.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source data and print counts without copying files or writing meta.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_tn3k_to_mvtec(
        src_root=args.src,
        dst_root=args.dst,
        fold_json=args.fold_json,
        class_name=args.class_name,
        defect_name=args.defect_name,
        train_source=args.train_source,
        test_source=args.test_source,
        mask_threshold=args.mask_threshold,
        validate_size=not args.skip_size_validation,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Prepared"
    print(f"{action} TN3K MVTec layout at {args.dst}")


if __name__ == "__main__":
    main()
