import torch

from backbones.CLIP import CLIPTextEncoder
from tqdm import tqdm
import os

def prompt_generator(dataset):
    class_names = dataset.get_class_names()
    real_names = dataset.get_real_names()
    prompts = dataset.get_prompts()

    generated_prompts = {'normal': {}, 'abnormal': {}}
    for template in prompts["prompt_templates"]:
        for class_name in class_names:
            real_name = real_names[class_name]
            generated_prompts['normal'][class_name] = []
            generated_prompts['abnormal'][class_name] = []
            for normal_template in prompts["prompt_normal"]:
                generated_prompts['normal'][class_name].append(template.format(normal_template.format(real_name)))
            for abnormal_template in prompts["prompt_abnormal"]:
                generated_prompts['abnormal'][class_name].append(template.format(abnormal_template.format(real_name)))
    
    return generated_prompts

def generate_clip_text_embeddings(args, dataset):
    prompts = prompt_generator(dataset)
    clip_text_encoder = CLIPTextEncoder(args.model_id, args.device)

    embeddings = {}
    
    total = sum(1 for state in prompts.keys() for class_name in prompts[state].keys())
    tqdm_bar = tqdm(total=total, desc="Encoding prompts with CLIP")
    for state in prompts.keys():
        embeddings[state] = {}
        for class_name in prompts[state].keys():
            embeddings[state][class_name] = clip_text_encoder(prompts[state][class_name])
            tqdm_bar.update(1)
            tqdm_bar.set_postfix({"State": state, "Class": class_name})
    
    return embeddings

def save_model(model, path, epoch):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(path, f'model_epoch_{epoch}.pth'))