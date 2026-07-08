import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import timeit
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DINOImageEncoder = None


def get_dino_encoder_class():
    global DINOImageEncoder
    if DINOImageEncoder is None:
        from backbones.DINO import DINOImageEncoder as _DINOImageEncoder
        DINOImageEncoder = _DINOImageEncoder
    return DINOImageEncoder
DATASET_CONSTANTS_CLASS = None


def get_dataset_constants_class():
    global DATASET_CONSTANTS_CLASS
    if DATASET_CONSTANTS_CLASS is None:
        constants_path = ROOT / "datasets" / "constants.py"
        spec = importlib.util.spec_from_file_location("zsad_dataset_constants", constants_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DATASET_CONSTANTS_CLASS = module.DatasetConstants
    return DATASET_CONSTANTS_CLASS

from model.model import ZSADModel
from utils.args import Args


DEFAULT_ARGS = {
    "model_id": "openai/clip-vit-large-patch14-336",
    "vision_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vision_layers": [6, 12, 18, 24],
    "base_dir": "./",
    "dataset_name": "mvtec",
    "mode": "test",
    "batch_size": 32,
    "img_size": 768,
    "lr": 1e-4,
    "start_epochs": 0,
    "end_epochs": 1,
    "output_dir": "./checkpoints",
    "out_dim": 768,
    "global_topk_ratio": 0.01,
}


def parse_vision_layers(value):
    if isinstance(value, (list, tuple)):
        return [int(layer) for layer in value]
    return [int(layer.strip()) for layer in str(value).split(",") if layer.strip()]


def resolve_device(requested_device, explicit=False):
    if requested_device is None:
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if explicit:
            raise RuntimeError(f"Requested device '{requested_device}', but CUDA is not available.")
        print(f"CUDA device '{requested_device}' is not available; falling back to CPU.")
        return torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def checkpoint_args_path(checkpoint, explicit_path):
    if explicit_path is not None:
        return Path(explicit_path)
    if checkpoint is None:
        return None
    return Path(checkpoint).parent / "args.json"


def build_model_args(cli_args):
    config = dict(DEFAULT_ARGS)
    checkpoint = Path(cli_args.checkpoint) if cli_args.checkpoint else None
    args_path = checkpoint_args_path(checkpoint, cli_args.checkpoint_args)

    if args_path is not None:
        if args_path.exists():
            config.update(load_json(args_path))
        elif cli_args.checkpoint_args is not None:
            raise FileNotFoundError(f"Checkpoint args file not found: {args_path}")
        else:
            print(f"No args.json found next to checkpoint; using perf defaults: {args_path}")

    overrides = {
        "dataset_name": cli_args.dataset_name,
        "model_id": cli_args.model_id,
        "vision_model_id": cli_args.vision_model_id,
        "vision_layers": cli_args.vision_layers,
        "img_size": cli_args.img_size,
        "out_dim": cli_args.out_dim,
        "global_topk_ratio": cli_args.global_topk_ratio,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if checkpoint is not None:
        config["output_dir"] = str(checkpoint.parent)

    config["vision_layers"] = parse_vision_layers(config["vision_layers"])
    device = resolve_device(cli_args.device or config.get("device"), explicit=cli_args.device is not None)
    config["device"] = str(device)

    return Args(**config), device


def prompt_counts(args, normal_override=None, abnormal_override=None):
    if normal_override is not None and abnormal_override is not None:
        return normal_override, abnormal_override

    dataset_constants = get_dataset_constants_class()
    prompts = dataset_constants(args.base_dir, args.dataset_name).get_prompts()
    templates = len(prompts["prompt_templates"])
    normal = templates * len(prompts["prompt_normal"])
    abnormal = templates * len(prompts["prompt_abnormal"])

    if normal_override is not None:
        normal = normal_override
    if abnormal_override is not None:
        abnormal = abnormal_override
    return normal, abnormal


def normalize_features(features):
    return features / (features.norm(dim=-1, keepdim=True) + 1e-8)


def prepare_text_embeddings(batch_size, args, device, normal_count, abnormal_count):
    normal = torch.randn(batch_size, normal_count, 768, device=device)
    abnormal = torch.randn(batch_size, abnormal_count, 768, device=device)
    return normalize_features(normal), normalize_features(abnormal)


def prepare_synthetic_features(batch_size, args, device):
    if args.img_size % 16 != 0:
        raise ValueError(f"img_size must be divisible by 16 for DINO patch features, got {args.img_size}.")

    patch_side = args.img_size // 16
    patch_count = patch_side * patch_side
    layer_count = len(args.vision_layers)
    cls = torch.randn(batch_size, layer_count, 1024, device=device)
    patches = torch.randn(batch_size, layer_count, patch_count, 1024, device=device)
    return cls, patches


def prepare_batch(batch_size, args, device, model_only, normal_count, abnormal_count):
    batch = {
        "text_embeddings": prepare_text_embeddings(batch_size, args, device, normal_count, abnormal_count),
    }
    if model_only:
        batch["features"] = prepare_synthetic_features(batch_size, args, device)
    else:
        batch["image"] = torch.randn(batch_size, 3, args.img_size, args.img_size, dtype=torch.float32)
    return batch


@torch.inference_mode()
def inference(batch, model, image_encoder, model_only):
    if model_only:
        cls, patches = batch["features"]
    else:
        cls, patches = image_encoder(batch["image"])

    anomaly_map, image_logits = model(batch["text_embeddings"], [cls, patches])
    return anomaly_map, image_logits


def count_parameters(module):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def params(model, image_encoder=None):
    model_total, model_trainable = count_parameters(model)
    print("Total params ZSAD:", model_total, "Trainable params ZSAD:", model_trainable)

    all_total = model_total
    all_trainable = model_trainable
    if image_encoder is not None:
        dino_total, dino_trainable = count_parameters(image_encoder.model)
        print("Total params DINO:", dino_total, "Trainable params DINO:", dino_trainable)
        all_total += dino_total
        all_trainable += dino_trainable

    print("Total params (measured modules):", all_total, "Trainable params (measured modules):", all_trainable)


def load_checkpoint(model, checkpoint, device):
    if checkpoint is None:
        print("No checkpoint provided; benchmarking randomly initialized ZSAD weights.")
        return

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint}")


def prepare_model(args, checkpoint=None, model_only=False):
    model = ZSADModel(args).to(args.device)
    load_checkpoint(model, checkpoint, args.device)
    model.eval()

    image_encoder = None
    if not model_only:
        image_encoder = get_dino_encoder_class()(args.vision_model_id, args.vision_layers, device=args.device)
        for param in image_encoder.model.parameters():
            param.requires_grad_(False)
        image_encoder.model.eval()

    return model, image_encoder


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_cleanup(device):
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


@torch.inference_mode()
def inference_speed(model, image_encoder, args, device, model_only, normal_count, abnormal_count, reps=100, warmup_reps=10):
    batch = prepare_batch(1, args, device, model_only, normal_count, abnormal_count)

    for _ in tqdm(range(warmup_reps), desc="Warmup latency"):
        inference(batch, model, image_encoder, model_only)
    synchronize(device)

    total_time = 0.0
    for _ in tqdm(range(reps), desc="Timing latency"):
        synchronize(device)
        start = timeit.default_timer()
        inference(batch, model, image_encoder, model_only)
        synchronize(device)
        total_time += timeit.default_timer() - start

    ms = total_time * 1000.0 / reps
    print("Speed in ms:", ms)
    return ms


@torch.inference_mode()
def throughput(
    model,
    image_encoder,
    args,
    device,
    model_only,
    normal_count,
    abnormal_count,
    batch_size=16,
    reps=100,
    warmup_reps=10,
):
    batch = prepare_batch(batch_size, args, device, model_only, normal_count, abnormal_count)

    for _ in tqdm(range(warmup_reps), desc="Warmup throughput"):
        inference(batch, model, image_encoder, model_only)
    synchronize(device)

    total_time = 0.0
    for _ in tqdm(range(reps), desc="Timing throughput"):
        synchronize(device)
        start = timeit.default_timer()
        inference(batch, model, image_encoder, model_only)
        synchronize(device)
        total_time += timeit.default_timer() - start

    images_per_second = batch_size * reps / total_time
    print("Throughput:", images_per_second)
    return images_per_second


@torch.inference_mode()
def memory(model, image_encoder, args, device, model_only, normal_count, abnormal_count, reps=100, warmup_reps=10):
    if device.type != "cuda":
        print("Memory in MB: nan (CUDA memory stats are unavailable on CPU)")
        return math.nan

    batch = prepare_batch(1, args, device, model_only, normal_count, abnormal_count)

    for _ in tqdm(range(warmup_reps), desc="Warmup memory"):
        cuda_memory_cleanup(device)
        inference(batch, model, image_encoder, model_only)
        synchronize(device)

    total_memory = 0
    for _ in tqdm(range(reps), desc="Memory calc"):
        cuda_memory_cleanup(device)
        inference(batch, model, image_encoder, model_only)
        synchronize(device)
        total_memory += torch.cuda.max_memory_reserved(device)

    mbs = total_memory / 1e6 / reps
    print("Memory in MB:", mbs)
    return mbs


@torch.inference_mode()
def flops(model, image_encoder, args, device, model_only, normal_count, abnormal_count):
    batch = prepare_batch(1, args, device, model_only, normal_count, abnormal_count)
    inference(batch, model, image_encoder, model_only)
    synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        inference(batch, model, image_encoder, model_only)
        synchronize(device)

    gflops = sum(event.flops for event in prof.key_averages()) / 1e9
    print("GFLOPs:", gflops)
    return gflops


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ZSAD inference performance.")
    parser.add_argument("label", nargs="?", default="zsad", help="Label used in the default perf_<label>.csv filename.")
    parser.add_argument("--checkpoint", default=None, help="Optional ZSAD checkpoint to load.")
    parser.add_argument("--checkpoint-args", default=None, help="Optional args.json path. Defaults to the checkpoint sibling.")
    parser.add_argument("--output", default=None, help="CSV output path. Defaults to perf_<label>.csv.")
    parser.add_argument("--device", default=None, help="Device to benchmark on, for example cuda:0 or cpu.")
    parser.add_argument("--cycles", type=int, default=6, help="Number of benchmark cycles.")
    parser.add_argument("--reps", type=int, default=100, help="Timed repetitions per cycle.")
    parser.add_argument("--warmup-reps", type=int, default=10, help="Warmup repetitions before each timed section.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size used for the throughput benchmark.")
    parser.add_argument("--model-only", action="store_true", help="Benchmark ZSAD with synthetic DINO features instead of running DINO.")
    parser.add_argument("--keep-first-cycle", action="store_true", help="Write the first cycle to CSV instead of treating it as warmup.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name used only to infer prompt counts.")
    parser.add_argument("--model-id", default=None, help="CLIP text model id recorded in args; the text encoder is not run.")
    parser.add_argument("--vision-model-id", default=None, help="DINO model id for the image encoder.")
    parser.add_argument("--vision-layers", default=None, help="Comma-separated DINO hidden-state layers, for example 6,12,18,24.")
    parser.add_argument("--img-size", type=int, default=None, help="Synthetic image size.")
    parser.add_argument("--out-dim", type=int, default=None, help="ZSAD adapter output dimension.")
    parser.add_argument("--global-topk-ratio", type=float, default=None, help="Top patch ratio for global image scoring.")
    parser.add_argument("--normal-prompts", type=int, default=None, help="Override the number of normal text embeddings.")
    parser.add_argument("--abnormal-prompts", type=int, default=None, help="Override the number of abnormal text embeddings.")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    if cli_args.cycles < 1:
        raise ValueError("--cycles must be >= 1.")
    if cli_args.reps < 1:
        raise ValueError("--reps must be >= 1.")
    if cli_args.warmup_reps < 0:
        raise ValueError("--warmup-reps must be >= 0.")
    if cli_args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    args, device = build_model_args(cli_args)
    normal_count, abnormal_count = prompt_counts(args, cli_args.normal_prompts, cli_args.abnormal_prompts)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(args)
    print(f"Prompt embeddings per image: normal={normal_count}, abnormal={abnormal_count}")
    print("Benchmark mode:", "ZSAD only" if cli_args.model_only else "DINO + ZSAD")

    output_path = Path(cli_args.output) if cli_args.output else Path(f"perf_{cli_args.label}.csv")
    model, image_encoder = prepare_model(args, cli_args.checkpoint, cli_args.model_only)
    params(model, image_encoder)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["time_ms", "throughput_img_s", "memory_mb", "gflops"])

        for cycle in range(cli_args.cycles):
            print(f"Cycle {cycle + 1}/{cli_args.cycles}")
            ms = inference_speed(
                model,
                image_encoder,
                args,
                device,
                cli_args.model_only,
                normal_count,
                abnormal_count,
                reps=cli_args.reps,
                warmup_reps=cli_args.warmup_reps,
            )
            images_per_second = throughput(
                model,
                image_encoder,
                args,
                device,
                cli_args.model_only,
                normal_count,
                abnormal_count,
                batch_size=cli_args.batch_size,
                reps=cli_args.reps,
                warmup_reps=cli_args.warmup_reps,
            )
            mbs = memory(
                model,
                image_encoder,
                args,
                device,
                cli_args.model_only,
                normal_count,
                abnormal_count,
                reps=cli_args.reps,
                warmup_reps=cli_args.warmup_reps,
            )
            gflops = flops(model, image_encoder, args, device, cli_args.model_only, normal_count, abnormal_count)

            if cycle == 0 and cli_args.cycles > 1 and not cli_args.keep_first_cycle:
                print("Skipping first cycle in CSV warmup.")
                continue

            writer.writerow([ms, images_per_second, mbs, gflops])
            csv_file.flush()
            print("-" * 42)
            print("Speed [ms]:", ms)
            print("Throughput [img/s]:", images_per_second)
            print("Memory [MB]:", mbs)
            print("GFLOPs:", gflops)
            print("-" * 42)

    print(f"Wrote benchmark CSV: {output_path}")


if __name__ == "__main__":
    main()
