import json

import torch

from backbones.CLIP import CLIPTextEncoder
from tqdm import tqdm
import os
import numpy as np

def prompt_generator(dataset):
    class_names = dataset.get_class_names()
    real_names = dataset.get_real_names()
    prompts = dataset.get_prompts()

    generated_prompts = {'normal': {}, 'abnormal': {}}

    # initialize once per class
    for class_name in class_names:
        generated_prompts['normal'][class_name] = []
        generated_prompts['abnormal'][class_name] = []

    # then append prompts
    for template in prompts["prompt_templates"]:
        for class_name in class_names:
            real_name = real_names[class_name]

            for normal_template in prompts["prompt_normal"]:
                text = normal_template.format(real_name)
                generated_prompts['normal'][class_name].append(
                    template.format(text)
                )

            for abnormal_template in prompts["prompt_abnormal"]:
                text = abnormal_template.format(real_name)
                generated_prompts['abnormal'][class_name].append(
                    template.format(text)
                )

    # with open(f'{dataset.dataset_name}_generated_prompts.json', 'w') as f:
    #     json.dump(generated_prompts, f, indent=4)

    return generated_prompts

def generate_clip_text_embeddings(args, dataset):
    prompts = prompt_generator(dataset)
    clip_text_encoder = CLIPTextEncoder(args.model_id, args.device)

    embeddings = {}
    
    total = sum(1 for state in prompts.keys() for _ in prompts[state].keys())
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

def fast_auc_and_best_f1(y_true, y_score):
    """
    Exact AUROC + best F1 using one sort only.
    y_true: binary array (0/1)
    y_score: continuous scores
    """
    y_true = np.asarray(y_true, dtype=np.uint8).ravel()
    y_score = np.asarray(y_score, dtype=np.float32).ravel()

    n = y_true.size
    n_pos = int(y_true.sum())
    n_neg = n - n_pos

    # Edge cases
    if n_pos == 0:
        return float("nan"), 0.0
    if n_neg == 0:
        return float("nan"), 1.0

    # Sort once by descending score
    order = np.argsort(-y_score, kind="mergesort")
    y_score_sorted = y_score[order]
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted, dtype=np.int64)
    fp = np.cumsum(1 - y_true_sorted, dtype=np.int64)

    # Keep only points where threshold changes
    distinct = np.r_[y_score_sorted[1:] != y_score_sorted[:-1], True]
    idx = np.flatnonzero(distinct)

    tp = tp[idx]
    fp = fp[idx]

    # ----- Best F1 -----
    precision = tp / (tp + fp)
    recall = tp / n_pos
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    best_f1 = float(f1.max()) if f1.size > 0 else 0.0

    # ----- AUROC -----
    tpr = tp / n_pos
    fpr = fp / n_neg

    # prepend origin
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]

    auroc = float(np.trapezoid(tpr, fpr))
    return auroc, best_f1