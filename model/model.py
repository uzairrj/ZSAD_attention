from torch.nn import Linear, Module, ModuleList, functional as F
from .adapters import Adapter
import torch
from .attention import CrossAttention

class ZSADModel(Module):
    def __init__(self, args):
        super(ZSADModel, self).__init__()

        self.number_of_vision_layers = len(args.vision_layers)

        self.positive_text_adapter = Adapter(768, args.out_dim, bottleneck=None, last_activation=True)
        self.negative_text_adapter = Adapter(768, args.out_dim, bottleneck=None, last_activation=True)
        self.patch_adapter = ModuleList([Adapter(1024, args.out_dim, bottleneck=None, last_activation=True)
                              for _ in range(self.number_of_vision_layers)])
        self.cls_adapter = ModuleList([Adapter(1024, args.out_dim, bottleneck=None, last_activation=True)
                              for _ in range(self.number_of_vision_layers)])

        self.cross_model_contrastive_learning_attention = ModuleList([CrossAttention(args.out_dim, 1, 0.2)
                                                                      for _ in range(self.number_of_vision_layers)])

        self.anomaly_aware_calibration_attention = ModuleList([CrossAttention(args.out_dim, 1, 0.2)
                                                                      for _ in range(self.number_of_vision_layers)])

        self.global_anomaly_alignment_attention = ModuleList([CrossAttention(args.out_dim, 1, 0.2)
                                                                      for _ in range(self.number_of_vision_layers)])
        self.cross_model_contrastive_head = ModuleList([Linear(args.out_dim, 1)
                                                        for _ in range(self.number_of_vision_layers)])
        self.anomaly_aware_calibration_head = ModuleList([Linear(args.out_dim, 1)
                                                          for _ in range(self.number_of_vision_layers)])
        self.global_anomaly_alignment_head = ModuleList([Linear(args.out_dim, 1)
                                                         for _ in range(self.number_of_vision_layers)])

        self.img_size = args.img_size

    def patch_logits_to_map(self, patch_logits):
        batch_size, patch_count, channels = patch_logits.shape
        grid_size = int(patch_count ** 0.5)
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Expected square patch grid, got {patch_count} patches")
        return patch_logits.permute(0, 2, 1).contiguous().view(batch_size, channels, grid_size, grid_size)

    def unpack_text_embeddings(self, text_embeddings):
        if isinstance(text_embeddings, (tuple, list)):
            return text_embeddings
        if text_embeddings.dim() == 4:
            return text_embeddings[:, 0], text_embeddings[:, 1]
        return text_embeddings[:, 0].unsqueeze(1), text_embeddings[:, 1].unsqueeze(1)

    def merge_text_embeddings(self, normal_text_adapted, abnormal_text_adapted):
        return torch.cat([normal_text_adapted, abnormal_text_adapted], dim=1)

    def cross_model_contrastive_learning(self, text_embeddings_adapted, patch_embeddings_adapted, layer_idx):
        patch_features = self.cross_model_contrastive_learning_attention[layer_idx](
            patch_embeddings_adapted,
            text_embeddings_adapted,
        )
        anomaly_map_cross_modal = self.cross_model_contrastive_head[layer_idx](patch_features)
        anomaly_map_cross_modal = F.interpolate(self.patch_logits_to_map(anomaly_map_cross_modal),
                                    size=self.img_size, mode='bilinear', align_corners=True)
        return anomaly_map_cross_modal

    def anomaly_aware_calibration(self, adapted_patch_features, cls_embeddings_adapted, layer_idx):
        anomaly_awareness_features = self.anomaly_aware_calibration_attention[layer_idx](
            adapted_patch_features,
            cls_embeddings_adapted,
        )
        anomaly_awareness_cls_patch = self.anomaly_aware_calibration_head[layer_idx](anomaly_awareness_features)
        anomaly_awareness_cls_patch = F.interpolate(self.patch_logits_to_map(anomaly_awareness_cls_patch),
                                    size=self.img_size, mode='bilinear', align_corners=True)
        return anomaly_awareness_cls_patch
    

    def global_anomaly_allignment(self, cls_embeddings_adapted, text_embeddings_adapted, layer_idx):
        cls_features = self.global_anomaly_alignment_attention[layer_idx](
            cls_embeddings_adapted,
            text_embeddings_adapted,
        )
        return self.global_anomaly_alignment_head[layer_idx](cls_features).squeeze(1)
    
    def forward(self, text_embeddings, image_features):
        normal_text_embeddings, abnormal_text_embeddings = self.unpack_text_embeddings(text_embeddings)
        positive_text_adapted = self.positive_text_adapter(normal_text_embeddings)
        positive_text_adapted = positive_text_adapted / (positive_text_adapted.norm(dim=-1, keepdim=True)+ 1e-8)
        negative_text_adapted = self.negative_text_adapter(abnormal_text_embeddings)
        negative_text_adapted = negative_text_adapted / (negative_text_adapted.norm(dim=-1, keepdim=True)+ 1e-8)
        adapted_text_embeddings = self.merge_text_embeddings(positive_text_adapted, negative_text_adapted)

        cls_embeddings_adapted = []
        for i in range(self.number_of_vision_layers):
            cls_embedding_adapted = self.cls_adapter[i](image_features[0][:, i, :])
            cls_embedding_adapted = cls_embedding_adapted / (cls_embedding_adapted.norm(dim=-1, keepdim=True)+ 1e-8)
            cls_embeddings_adapted.append(cls_embedding_adapted)

        patch_embeddings_adapted = []
        for i in range(self.number_of_vision_layers):
            patch_embedding_adapted = self.patch_adapter[i](image_features[1][:,i,:,:])
            patch_embedding_adapted = patch_embedding_adapted / (patch_embedding_adapted.norm(dim=-1, keepdim=True)+ 1e-8)
            patch_embeddings_adapted.append(patch_embedding_adapted)

        cross_model_contrastive = torch.stack([
            self.cross_model_contrastive_learning(adapted_text_embeddings, patch_embeddings_adapted[i], i)
            for i in range(self.number_of_vision_layers)], dim=1)

        # anomaly_aware_calibration = torch.stack([
        #     self.anomaly_aware_calibration(patch_embeddings_adapted[i], cls_embeddings_adapted[i].unsqueeze(1), i)
        #     for i in range(self.number_of_vision_layers)], dim=1)

        global_anomaly_alignment = torch.stack([
            self.global_anomaly_allignment(cls_embeddings_adapted[i].unsqueeze(1), adapted_text_embeddings, i)
            for i in range(self.number_of_vision_layers)], dim=1)

        return cross_model_contrastive.mean(dim=1), global_anomaly_alignment.mean(dim=1)
