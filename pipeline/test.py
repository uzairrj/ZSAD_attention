import os

from sklearn.metrics import roc_auc_score, precision_recall_curve
import torch
import json
from datasets.constants import DatasetConstants
from utils.utils import generate_clip_text_embeddings, fast_auc_and_best_f1
from tqdm import tqdm
from backbones.DINO import DINOImageEncoder
from torch.utils.data import DataLoader
from datasets import get_data
from utils.transformations import get_transforms
from model.model import ZSADModel
import numpy as np

def compute_best_f1(y_true, y_score):
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    f1_scores = f1_scores[np.isfinite(f1_scores)]
    return np.max(f1_scores) if len(f1_scores) > 0 else 0.0

def max_map_image_score(anomaly_maps):
    flat_maps = anomaly_maps.flatten(1)
    return flat_maps.max(dim=1).values

def testing_epoch(dataloader, model, image_encoder, text_embeddings):
    pixel_gt = []
    pixel_pred = []
    image_gt = []
    image_text_pred = []
    image_map_pred = []
    image_fused_pred = []
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

        cross_model_contrastive, text_branch_logits = model(batched_text_embeddings, [cls, patches])
        pixel_anomaly_map = torch.sigmoid(cross_model_contrastive[:, 0, :, :])
        text_image_score = torch.sigmoid(text_branch_logits).view(-1)
        map_image_score = max_map_image_score(pixel_anomaly_map)
        fused_image_score = text_image_score + map_image_score

        pixel_gt.extend(data['img_mask'].squeeze(1).cpu().detach().numpy())
        image_gt.extend(data['anomaly'].cpu().detach().numpy())
        img_list.extend(data["img_path"])
        pixel_pred.extend(pixel_anomaly_map.cpu().detach().numpy())
        image_text_pred.extend(text_image_score.cpu().detach().numpy())
        image_map_pred.extend(map_image_score.cpu().detach().numpy())
        image_fused_pred.extend(fused_image_score.cpu().detach().numpy())
        cls_list.extend(data['cls_name'])

    gt_mask_list = np.array(pixel_gt)   
    pred_mask_list = np.array(pixel_pred) 
    gt_image_list = np.array(image_gt)
    pred_image_text_list = np.array(image_text_pred)
    pred_image_map_list = np.array(image_map_pred)
    pred_image_fused_list = np.array(image_fused_pred)
    cls_list = np.array(cls_list)

    cls_list = np.asarray(cls_list)
    gt_mask_list = np.asarray(gt_mask_list)
    pred_mask_list = np.asarray(pred_mask_list, dtype=np.float32)
    gt_image_list = np.asarray(gt_image_list, dtype=np.uint8)
    pred_image_text_list = np.asarray(pred_image_text_list, dtype=np.float32)
    pred_image_map_list = np.asarray(pred_image_map_list, dtype=np.float32)
    pred_image_fused_list = np.asarray(pred_image_fused_list, dtype=np.float32)


    class_metrics = {}
    auroc_list = []
    f1_list = []
    image_auroc_list = []
    image_f1_list = []
    image_text_auroc_list = []
    image_text_f1_list = []
    image_map_auroc_list = []
    image_map_f1_list = []

    for cls_name in np.unique(cls_list):
        idx = (cls_list == cls_name)

        # ravel() avoids unnecessary copies when possible
        gt_cls = gt_mask_list[idx].ravel().astype(np.uint8, copy=False)
        pred_cls = pred_mask_list[idx].ravel().astype(np.float32, copy=False)
        gt_image_cls = gt_image_list[idx].astype(np.uint8, copy=False)
        pred_image_text_cls = pred_image_text_list[idx].astype(np.float32, copy=False)
        pred_image_map_cls = pred_image_map_list[idx].astype(np.float32, copy=False)
        pred_image_fused_cls = pred_image_fused_list[idx].astype(np.float32, copy=False)

        auroc, f1 = fast_auc_and_best_f1(gt_cls, pred_cls)
        image_text_auroc, image_text_f1 = fast_auc_and_best_f1(gt_image_cls, pred_image_text_cls)
        image_map_auroc, image_map_f1 = fast_auc_and_best_f1(gt_image_cls, pred_image_map_cls)
        image_auroc, image_f1 = fast_auc_and_best_f1(gt_image_cls, pred_image_fused_cls)

        class_metrics[cls_name] = {
            "Pixel_AUROC": auroc,
            "Pixel_F1": f1,
            "Image_AUROC": image_auroc,
            "Image_F1": image_f1,
            "Image_Text_AUROC": image_text_auroc,
            "Image_Text_F1": image_text_f1,
            "Image_Map_AUROC": image_map_auroc,
            "Image_Map_F1": image_map_f1,
        }

        if not np.isnan(auroc):
            auroc_list.append(auroc)
        f1_list.append(f1)
        if not np.isnan(image_auroc):
            image_auroc_list.append(image_auroc)
        image_f1_list.append(image_f1)
        if not np.isnan(image_text_auroc):
            image_text_auroc_list.append(image_text_auroc)
        image_text_f1_list.append(image_text_f1)
        if not np.isnan(image_map_auroc):
            image_map_auroc_list.append(image_map_auroc)
        image_map_f1_list.append(image_map_f1)

        print(
            f"{cls_name}: "
            f"AUC_Pixel={auroc:.5f}\tF1_Pixel={f1:.5f}\t"
            f"AUC_Image={image_auroc:.5f}\tF1_Image={image_f1:.5f}\t"
            f"AUC_Text={image_text_auroc:.5f}\tAUC_Map={image_map_auroc:.5f}"
        )

    mean_auroc = np.mean(auroc_list) if len(auroc_list) > 0 else float("nan")
    mean_f1 = np.mean(f1_list) if len(f1_list) > 0 else float("nan")
    mean_image_auroc = np.mean(image_auroc_list) if len(image_auroc_list) > 0 else float("nan")
    mean_image_f1 = np.mean(image_f1_list) if len(image_f1_list) > 0 else float("nan")
    mean_image_text_auroc = np.mean(image_text_auroc_list) if len(image_text_auroc_list) > 0 else float("nan")
    mean_image_text_f1 = np.mean(image_text_f1_list) if len(image_text_f1_list) > 0 else float("nan")
    mean_image_map_auroc = np.mean(image_map_auroc_list) if len(image_map_auroc_list) > 0 else float("nan")
    mean_image_map_f1 = np.mean(image_map_f1_list) if len(image_map_f1_list) > 0 else float("nan")

    print(f"\nMean AUC_Pixel: {mean_auroc:.5f}")
    print(f"Mean F1_Pixel:  {mean_f1:.5f}")
    print(f"Mean AUC_Image: {mean_image_auroc:.5f}")
    print(f"Mean F1_Image:  {mean_image_f1:.5f}")
    print(f"Mean AUC_Image_Text: {mean_image_text_auroc:.5f}")
    print(f"Mean F1_Image_Text:  {mean_image_text_f1:.5f}")
    print(f"Mean AUC_Image_Map: {mean_image_map_auroc:.5f}")
    print(f"Mean F1_Image_Map:  {mean_image_map_f1:.5f}")


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
        
        
