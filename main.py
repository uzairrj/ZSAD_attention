from utils.args import Args
from pipeline.train import training
from pipeline.test import testing
from argparse import ArgumentParser

def main(args):
    if args.mode == "train":
        training(args)
    if args.mode == "test":
        testing(args)

if __name__ == "__main__":

    args_parser = ArgumentParser(description="Zero-Shot Anomaly Detection with Attention")
    args_parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    args_parser.add_argument('--dataset_name', type=str, default='mvtec', help='Dataset name')
    args_parser.add_argument('--start_epochs', type=int, default=0, help='Starting epoch for training or testing')
    args_parser.add_argument('--end_epochs', type=int, default=20, help='Ending epoch for training or testing')
    args_parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training/testing')
    args_parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory for checkpoints and results')
    args_parser.add_argument('--global_topk_ratio', type=float, default=0.01, help='Top patch ratio for global patch-text contrast')

    args = args_parser.parse_args()

    CLIP_args = {
        "model_id": "openai/clip-vit-large-patch14-336"
    }

    DINO_args = {
        "vision_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        'vision_layers': [6,12,18,24]
    }

    generic_args = {
        "device": args.device,
        "base_dir": "./",
        "dataset_name": args.dataset_name,
        'mode': args.mode,
        'batch_size': 32,
        'img_size': 756,
        'lr': 1e-4,
        'start_epochs': args.start_epochs,
        'end_epochs': args.end_epochs,
        'output_dir': args.output_dir,
        'out_dim': 768,
        'global_topk_ratio': args.global_topk_ratio
    }


    args = Args(
        **CLIP_args,
        **DINO_args,
        **generic_args
    )

    print(args)

    main(args)
