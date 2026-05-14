import os

import torch
from datasets.constants import DatasetConstants
from utils.utils import generate_clip_text_embeddings, fast_auc_and_best_f1
from tqdm import tqdm
from backbones.DINO import DINOImageEncoder
from torch.utils.data import DataLoader
from datasets import get_data
from utils.transformations import get_transforms
from model.model import ZSADModel
import numpy as np

def testing_epoch(dataloader, model, image_encoder, text_embeddings):
    pixel_gt = []
    pixel_pred = []
    image_gt = []
    image_model_pred = []
    img_list = []
    cls_list = []
    for data in tqdm(dataloader):
        normal_list = []
        abnormal_list = []

        for cls in data['cls_name']:
            normal_list.append(text_embeddings['normal'][cls])
            abnormal_list.append(text_embeddings['abnormal'][cls])

        normal_batch = torch.stack(normal_list, dim=0)
        abnormal_batch = torch.stack(abnormal_list, dim=0)
        batched_text_embeddings = (normal_batch, abnormal_batch)
        
        cls, patches = image_encoder(data['img'])

        cross_model_contrastive, image_level_logits = model(batched_text_embeddings, [cls, patches])
        pixel_anomaly_map = torch.sigmoid(cross_model_contrastive[:, 0, :, :])
        image_model_score = torch.sigmoid(image_level_logits).view(-1)

        pixel_gt.extend(data['img_mask'].squeeze(1).cpu().detach().numpy())
        image_gt.extend(data['anomaly'].cpu().detach().numpy())
        img_list.extend(data["img_path"])
        pixel_pred.extend(pixel_anomaly_map.cpu().detach().numpy())
        image_model_pred.extend(image_model_score.cpu().detach().numpy())
        cls_list.extend(data['cls_name'])

    gt_mask_list = np.array(pixel_gt)   
    pred_mask_list = np.array(pixel_pred) 
    gt_image_list = np.array(image_gt)
    pred_image_model_list = np.array(image_model_pred)
    cls_list = np.array(cls_list)

    cls_list = np.asarray(cls_list)
    gt_mask_list = np.asarray(gt_mask_list)
    pred_mask_list = np.asarray(pred_mask_list, dtype=np.float32)
    gt_image_list = np.asarray(gt_image_list, dtype=np.uint8)
    pred_image_model_list = np.asarray(pred_image_model_list, dtype=np.float32)


    auroc_list = []
    f1_list = []
    image_auroc_list = []
    image_f1_list = []

    for cls_name in np.unique(cls_list):
        idx = (cls_list == cls_name)

        # ravel() avoids unnecessary copies when possible
        gt_cls = gt_mask_list[idx].ravel().astype(np.uint8, copy=False)
        pred_cls = pred_mask_list[idx].ravel().astype(np.float32, copy=False)
        gt_image_cls = gt_image_list[idx].astype(np.uint8, copy=False)
        pred_image_model_cls = pred_image_model_list[idx].astype(np.float32, copy=False)

        auroc, f1 = fast_auc_and_best_f1(gt_cls, pred_cls)
        image_auroc, image_f1 = fast_auc_and_best_f1(gt_image_cls, pred_image_model_cls)

        if not np.isnan(auroc):
            auroc_list.append(auroc)
        f1_list.append(f1)
        if not np.isnan(image_auroc):
            image_auroc_list.append(image_auroc)
        image_f1_list.append(image_f1)

        print(
            f"{cls_name}: "
            f"AUC_Pixel={auroc:.5f}\tF1_Pixel={f1:.5f}\t"
            f"AUC_Image={image_auroc:.5f}\tF1_Image={image_f1:.5f}"
        )

    mean_auroc = np.mean(auroc_list) if len(auroc_list) > 0 else float("nan")
    mean_f1 = np.mean(f1_list) if len(f1_list) > 0 else float("nan")
    mean_image_auroc = np.mean(image_auroc_list) if len(image_auroc_list) > 0 else float("nan")
    mean_image_f1 = np.mean(image_f1_list) if len(image_f1_list) > 0 else float("nan")

    print(f"\nMean AUC_Pixel: {mean_auroc:.5f}")
    print(f"Mean F1_Pixel:  {mean_f1:.5f}")
    print(f"Mean AUC_Image: {mean_image_auroc:.5f}")
    print(f"Mean F1_Image:  {mean_image_f1:.5f}")


def testing(args):
    dataset_constants = DatasetConstants(args.base_dir, args.dataset_name)

    # Cache CLIP text embeddings
    text_embeddings = generate_clip_text_embeddings(args, dataset_constants)

    transform_img, transform_mask = get_transforms(args.img_size)

    model = ZSADModel(args).to(args.device)
    image_encoder = DINOImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)

    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    for i in range(args.start_epochs, args.end_epochs):
        model_name = f'model_epoch_{i+1}.pth'
        print(f"Testing with model: {model_name}")
        model_path = os.path.join(args.output_dir, model_name)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        with torch.no_grad():
            testing_epoch(dataloader, model, image_encoder, text_embeddings)
        
        
