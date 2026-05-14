import os

import torch
import json
from datasets.constants import DatasetConstants
from utils.loss import BinaryDiceLoss, BinaryFocalLossWithLogits
from utils.utils import generate_clip_text_embeddings, save_model
from tqdm import tqdm
from backbones.DINO import DINOImageEncoder
from torch.utils.data import DataLoader
from datasets import get_data
from utils.transformations import get_transforms
from model.model import ZSADModel

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
        args.start_epochs += 1
    else:
        print("No pre-trained model found. Starting training from scratch.")
        args.start_epochs = 0

    image_encoder = DINOImageEncoder(args.vision_model_id, args.vision_layers, device=args.device)

    dataset = get_data(args.dataset_name, transform_img, transform_mask, training=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    loss_dice = BinaryDiceLoss()
    pixel_focal = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0)
    image_focal = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0)
    seg_loss_weight = 0.3
    text_loss_weight = 0.7

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    json.dump(vars(args), open(os.path.join(args.output_dir, f'args.json'), 'w'), indent=4)

    for epoch in range(args.start_epochs, args.end_epochs):
        losses = {
            'segmentation_loss': 0.0,
            'text_anomaly_loss': 0.0,
            'weighted_segmentation_loss': 0.0,
            'weighted_text_anomaly_loss': 0.0,
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
            batched_text_embeddings = (normal_batch, abnormal_batch)
            
            cls, patches = image_encoder(data['img'])

            cross_model_contrastive, global_cls_text_loss = model(batched_text_embeddings, [cls, patches])
            
            
            mask = data['img_mask'].to(args.device).float()

            # anomaly_awareness_prob = torch.sigmoid(anomaly_aware_calibration)
            cross_model_prob = torch.sigmoid(cross_model_contrastive)
            anomaly_targets = data['anomaly'].to(args.device).float().unsqueeze(1)

            # anomaly_awareness_loss = F.binary_cross_entropy_with_logits(anomaly_aware_calibration, mask) + loss_dice(anomaly_awareness_prob, mask)
            seg_loss = pixel_focal(cross_model_contrastive, mask) + loss_dice(cross_model_prob, mask)
            # global_cls_text_loss = F.sigmoid(global_anomaly_alignment) - anomaly
            text_anomaly_loss = image_focal(global_cls_text_loss, anomaly_targets)
            weighted_seg_loss = seg_loss_weight * seg_loss
            weighted_text_loss = text_loss_weight * text_anomaly_loss
            loss = weighted_seg_loss + weighted_text_loss

            # losses['anomaly_awareness_loss'] += anomaly_awareness_loss.item()
            losses['segmentation_loss'] += seg_loss.item()
            losses['text_anomaly_loss'] += text_anomaly_loss.item()
            losses['weighted_segmentation_loss'] += weighted_seg_loss.item()
            losses['weighted_text_anomaly_loss'] += weighted_text_loss.item()
            losses['total_loss'] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses = {k: v / len(dataloader) for k, v in losses.items()}
        print(
            f"Epoch {epoch+1}/{args.end_epochs}: "
            f"Segmentation Loss: {losses['segmentation_loss']:.4f}, "
            f"Text Anomaly Loss: {losses['text_anomaly_loss']:.4f}, "
            f"Weighted Segmentation: {losses['weighted_segmentation_loss']:.4f}, "
            f"Weighted Text: {losses['weighted_text_anomaly_loss']:.4f}, "
            f"Total Loss: {losses['total_loss']:.4f}",
            flush=True
        )
        save_model(model, args.output_dir, epoch+1)
