from datasets.constants import DatasetConstants
from utils.utils import prompt_generator, generate_clip_text_embeddings
from backbones.CLIP import CLIPTextEncoder
from tqdm import tqdm


def training(args):
    dataset = DatasetConstants(args.base_dir, args.dataset_name)

    # Cache CLIP text embeddings
    text_embeddings = generate_clip_text_embeddings(args, dataset)