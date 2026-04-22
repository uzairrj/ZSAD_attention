from utils.args import Args
from pipeline.train import training

def main(args):
    if args.mode == "train":
        training(args)

if __name__ == "__main__":

    CLIP_args = {
        "model_id": "openai/clip-vit-large-patch14-336"
    }

    generic_args = {
        "device": "cuda:0",
        "base_dir": "./",
        "dataset_name": "MVTec",
        'mode': "train"
    }


    args = Args(
        **CLIP_args,
        **generic_args
    )

    print(args)

    main(args)