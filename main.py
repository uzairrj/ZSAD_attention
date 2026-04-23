from utils.args import Args
from pipeline.train import training
from pipeline.test import testing

def main(args):
    if args.mode == "train":
        training(args)
    if args.mode == "test":
        testing(args)

if __name__ == "__main__":

    CLIP_args = {
        "model_id": "openai/clip-vit-large-patch14-336"
    }

    DINO_args = {
        "vision_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        'vision_layers': [5,11,17,23]
    }

    generic_args = {
        "device": "cuda:0",
        "base_dir": "./",
        "dataset_name": "visa",
        'mode': "test",
        'batch_size': 64,
        'img_size': 512,
        'lr': 1e-4,
        'epochs': 10,
        'output_dir': './checkpoints',
        'out_dim': 768
    }


    args = Args(
        **CLIP_args,
        **DINO_args,
        **generic_args
    )

    print(args)

    main(args)