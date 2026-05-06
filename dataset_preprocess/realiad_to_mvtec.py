#!/usr/bin/env python3
"""Convert Real-IAD ZIP archives into the MVTec AD directory layout.

Expected source layout:

    real-iad/
        realiad_1024/
            audiojack.zip
            bottle_cap.zip
            ...
        realiad_jsons.zip

The official JSON annotations define train/test splits, class names, anomaly
types, image paths, and mask paths. This converter extracts the ZIP members into
MVTec-style folders:

    audiojack/
        train/good/
        test/good/
        test/hs/
        ground_truth/hs/
    meta.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_REALIAD_ROOT = Path("/media/data/ukhan/data/computer_vision/real-iad/real-iad")
DEFAULT_IMAGE_ZIP_DIR = "realiad_1024"
DEFAULT_JSON_ZIP = "realiad_jsons.zip"
DEFAULT_JSON_GROUP = "realiad_jsons"
GOOD_NAME = "good"


@dataclass(frozen=True)
class Annotation:
    class_name: str
    phase: str
    specie_name: str
    image_member: str
    mask_member: str | None
    image_suffix: str
    anomaly: int


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "defect"


def posix_join(*parts: str) -> str:
    result = PurePosixPath(parts[0])
    for part in parts[1:]:
        result /= part
    return result.as_posix()


def list_class_jsons(json_zip: zipfile.ZipFile, json_group: str) -> list[str]:
    prefix = json_group.rstrip("/") + "/"
    names = [
        name
        for name in json_zip.namelist()
        if name.startswith(prefix) and name.endswith(".json") and not name.endswith("/")
    ]
    return sorted(names, key=lambda name: PurePosixPath(name).stem)


def load_annotations(
    src_root: Path,
    json_group: str,
    selected_classes: set[str] | None,
) -> dict[str, list[Annotation]]:
    json_path = src_root / DEFAULT_JSON_ZIP
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing Real-IAD JSON ZIP: {json_path}")

    annotations: dict[str, list[Annotation]] = {}
    with zipfile.ZipFile(json_path) as json_zip:
        class_jsons = list_class_jsons(json_zip, json_group)
        if not class_jsons:
            raise ValueError(f"No JSON annotations found under {json_group!r} in {json_path}")

        for json_name in class_jsons:
            class_name = PurePosixPath(json_name).stem
            if selected_classes is not None and class_name not in selected_classes:
                continue

            data = json.loads(json_zip.read(json_name))
            prefix = data.get("meta", {}).get("prefix", f"{class_name}/")
            normal_class = data.get("meta", {}).get("normal_class", "OK")
            class_items: list[Annotation] = []

            for phase in ("train", "test"):
                for item in data.get(phase, []):
                    anomaly_class = item["anomaly_class"]
                    is_good = anomaly_class == normal_class
                    if phase == "train" and not is_good:
                        raise ValueError(
                            f"{class_name}: train split contains anomaly {item['image_path']}"
                        )

                    specie_name = GOOD_NAME if is_good else sanitize_name(anomaly_class)
                    mask_path = item.get("mask_path")
                    if not is_good and not mask_path:
                        raise ValueError(f"{class_name}: anomalous item is missing mask_path")

                    image_member = posix_join(prefix, item["image_path"])
                    mask_member = posix_join(prefix, mask_path) if mask_path else None
                    image_suffix = PurePosixPath(item["image_path"]).suffix.lower() or ".jpg"
                    class_items.append(
                        Annotation(
                            class_name=class_name,
                            phase=phase,
                            specie_name=specie_name,
                            image_member=image_member,
                            mask_member=mask_member,
                            image_suffix=image_suffix,
                            anomaly=0 if is_good else 1,
                        )
                    )

            annotations[class_name] = class_items

    if selected_classes is not None:
        missing = sorted(selected_classes - set(annotations))
        if missing:
            raise ValueError(f"Requested classes not found in JSON annotations: {', '.join(missing)}")

    return annotations


def validate_zip_members(
    src_root: Path,
    annotations: dict[str, list[Annotation]],
    image_zip_dir: str,
) -> None:
    for class_name, items in annotations.items():
        zip_path = src_root / image_zip_dir / f"{class_name}.zip"
        if not zip_path.is_file():
            raise FileNotFoundError(f"Missing Real-IAD class ZIP: {zip_path}")

        with zipfile.ZipFile(zip_path) as class_zip:
            members = set(class_zip.namelist())
            missing: list[str] = []
            for item in items:
                if item.image_member not in members:
                    missing.append(item.image_member)
                if item.mask_member is not None and item.mask_member not in members:
                    missing.append(item.mask_member)
                if len(missing) >= 10:
                    break
            if missing:
                details = ", ".join(missing[:10])
                raise FileNotFoundError(f"{class_name}: missing ZIP members: {details}")


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


def extract_member(
    class_zip: zipfile.ZipFile,
    member: str,
    dst_path: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with class_zip.open(member) as src, dst_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def make_meta_record(
    item: Annotation,
    rel_img: Path,
    rel_mask: Path | None,
) -> dict[str, object]:
    return {
        "img_path": rel_img.as_posix(),
        "mask_path": rel_mask.as_posix() if rel_mask is not None else "",
        "cls_name": item.class_name,
        "specie_name": item.specie_name,
        "anomaly": item.anomaly,
    }


def write_class(
    src_root: Path,
    dst_root: Path,
    class_name: str,
    items: list[Annotation],
    image_zip_dir: str,
    dry_run: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_info: list[dict[str, object]] = []
    test_info: list[dict[str, object]] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    zip_path = src_root / image_zip_dir / f"{class_name}.zip"

    with zipfile.ZipFile(zip_path) as class_zip:
        for item in items:
            phase = item.phase
            specie_name = item.specie_name
            index = counters[(phase, specie_name)]
            counters[(phase, specie_name)] += 1

            image_name = f"{index:06d}{item.image_suffix}"
            rel_img = Path(class_name) / phase / specie_name / image_name
            extract_member(class_zip, item.image_member, dst_root / rel_img, dry_run)

            rel_mask: Path | None = None
            if item.mask_member is not None:
                rel_mask = (
                    Path(class_name)
                    / "ground_truth"
                    / specie_name
                    / f"{index:06d}_mask.png"
                )
                extract_member(class_zip, item.mask_member, dst_root / rel_mask, dry_run)

            record = make_meta_record(item, rel_img, rel_mask)
            if phase == "train":
                train_info.append(record)
            else:
                test_info.append(record)

    return train_info, test_info


def convert_realiad_to_mvtec(
    src_root: Path,
    dst_root: Path,
    image_zip_dir: str = DEFAULT_IMAGE_ZIP_DIR,
    json_group: str = DEFAULT_JSON_GROUP,
    classes: list[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    selected_classes = set(classes) if classes else None
    annotations = load_annotations(src_root, json_group, selected_classes)
    class_names = sorted(annotations)

    validate_zip_members(src_root, annotations, image_zip_dir)
    if output_exists(dst_root, class_names):
        if not overwrite and not dry_run:
            raise FileExistsError(
                f"MVTec output already exists under {dst_root}. "
                "Use --overwrite to replace generated train/test/ground_truth folders and meta.json."
            )
        if overwrite and not dry_run:
            clean_output(dst_root, class_names)

    meta: dict[str, dict[str, list[dict[str, object]]]] = {"train": {}, "test": {}}
    total_train = total_test_good = total_test_bad = 0
    for class_name in class_names:
        train_info, test_info = write_class(
            src_root=src_root,
            dst_root=dst_root,
            class_name=class_name,
            items=annotations[class_name],
            image_zip_dir=image_zip_dir,
            dry_run=dry_run,
        )
        meta["train"][class_name] = train_info
        meta["test"][class_name] = test_info

        train_count = len(train_info)
        test_good = sum(1 for item in test_info if item["anomaly"] == 0)
        test_bad = sum(1 for item in test_info if item["anomaly"] == 1)
        total_train += train_count
        total_test_good += test_good
        total_test_bad += test_bad
        print(
            f"{class_name}: train/good={train_count} "
            f"test/good={test_good} test/anomaly={test_bad}"
        )

    if not dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)
        with (dst_root / "meta.json").open("w", encoding="utf-8") as file:
            file.write(json.dumps(meta, indent=4) + "\n")

    print(
        f"Total: train/good={total_train} "
        f"test/good={total_test_good} test/anomaly={total_test_bad}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Real-IAD ZIP archives into the MVTec AD folder format."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DEFAULT_REALIAD_ROOT,
        help=f"Real-IAD root containing realiad_1024 and realiad_jsons.zip. Default: {DEFAULT_REALIAD_ROOT}",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Converted output root. Default: same as --src, adding class folders in-place.",
    )
    parser.add_argument(
        "--image-zip-dir",
        default=DEFAULT_IMAGE_ZIP_DIR,
        help=f"Directory under --src that contains per-class ZIP files. Default: {DEFAULT_IMAGE_ZIP_DIR}",
    )
    parser.add_argument(
        "--json-group",
        default=DEFAULT_JSON_GROUP,
        help=f"Annotation folder inside realiad_jsons.zip. Default: {DEFAULT_JSON_GROUP}",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional subset of class names to convert, e.g. audiojack pcb zipper.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated train/test/ground_truth folders and meta.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSON/ZIP pairings and print counts without extracting files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dst = args.dst if args.dst is not None else args.src
    convert_realiad_to_mvtec(
        src_root=args.src,
        dst_root=dst,
        image_zip_dir=args.image_zip_dir,
        json_group=args.json_group,
        classes=args.classes,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Converted"
    print(f"{action} Real-IAD MVTec layout at {dst}")


if __name__ == "__main__":
    main()
