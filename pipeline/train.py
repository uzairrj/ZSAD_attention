import os

import torch
import json
from datasets.constants import DatasetConstants
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.utils import generate_clip_text_embeddings, save_model
from tqdm import tqdm
from backbones.DINO import DINOImageEncoder
from torch.utils.data import DataLoader
from datasets import get_data
from utils.transformations import get_transforms
from model.model import ZSADModel
from torch.nn import functional as F

def training(args):
    dataset_constants = DatasetConstants(args.base_dir, args.dataset_name)

    # Cache CLIP text embeddings
    text_embeddings = generate_clip_text_embeddings(args, dataset_constants)

    transform_img, transform_mask = get_transforms(args.img_size)

    model = ZSADModel(args).to(args.device)

    if args.start_epochs > 0 and os.path.exists(os.path.join(args.output_dir, f'model_epoch_{args.start_epochs}.pth')):
        model_path = os.path.join(args.output_dir, f'model_epoch_{args.start_epochs}.pth')
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No pre-trained model found. Starting training from scratch.")
        args.start_epochs = 0

    image_encoder = DINOImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)

    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    json.dump(vars(args), open(os.path.join(args.output_dir, f'args.json'), 'w'), indent=4)

    for epoch in range(args.start_epochs+1, args.end_epochs):
        losses = {
            'anomaly_awareness_loss': 0.0,
            'segmentation_loss': 0.0,
            'global_anomaly_loss': 0.0,
            'total_loss': 0.0
        }
        for data in tqdm(dataloader):
            normal_list = []
            abnormal_list = []

            for cls in data['cls_name']:
                normal_list.append(text_embeddings['normal'][cls])
                abnormal_list.append(text_embeddings['abnormal'][cls])

            normal_batch = torch.stack(normal_list, dim=0)
            abnormal_batch = torch.stack(abnormal_list, dim=0)
            batched_text_embeddings = torch.stack([normal_batch, abnormal_batch], dim=1)
            
            cls, patches = image_encoder(data['img'])

            cross_model_contrastive, anomaly_aware_calibration, global_anomaly_alignment = model(batched_text_embeddings, [cls, patches])
            
            
            mask = data['img_mask'].to(args.device)

            anomaly_awareness_loss = loss_focal(anomaly_aware_calibration, mask) + loss_dice(anomaly_aware_calibration[:, 1, :, :], mask)
            seg_loss = loss_focal(cross_model_contrastive, mask) + loss_dice(cross_model_contrastive[:, 1, :, :], mask)
            global_anomaly_loss = F.cross_entropy(global_anomaly_alignment.squeeze(1), data['anomaly'].to(args.device).long())
            loss = 0.25 * anomaly_awareness_loss + 0.5 * seg_loss + 0.25 * global_anomaly_loss

            losses['anomaly_awareness_loss'] += anomaly_awareness_loss.item()
            losses['segmentation_loss'] += seg_loss.item()
            losses['global_anomaly_loss'] += global_anomaly_loss.item()
            losses['total_loss'] += loss.item()

            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses = {k: v / len(dataloader) for k, v in losses.items()}
        print(f"Epoch {epoch+1}/{args.end_epochs}:Anomaly Awareness Loss: {losses['anomaly_awareness_loss']/len(dataloader):.4f}, Segmentation Loss: {losses['segmentation_loss']/len(dataloader):.4f}, Global Anomaly Loss: {losses['global_anomaly_loss']/len(dataloader):.4f}, Total Loss: {losses['total_loss']/len(dataloader):.4f}", flush=True)
        save_model(model, args.output_dir, epoch+1)
