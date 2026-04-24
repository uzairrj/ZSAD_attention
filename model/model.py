from torch.nn import Module, ModuleList, functional as F
from .adapters import Adapter
import torch

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
        
        self.img_size = args.img_size
        
    def cross_model_contrastive_learning(self, adapted_text_embeddings,patch_embeddings_adapted):
        anomaly_map_cross_modal = 100 * (patch_embeddings_adapted @ adapted_text_embeddings)
        anomaly_map_cross_modal = F.interpolate(anomaly_map_cross_modal.permute(0, 2, 1).view(-1, 2, 32, 32),
                                    size=self.img_size, mode='bilinear', align_corners=True)
        return torch.softmax(anomaly_map_cross_modal, dim=1)

    def anomaly_aware_calibration(self, adapted_patch_features, cls_embeddings_adapted):
        anomaly_awareness_cls_patch = 10 * (adapted_patch_features @ cls_embeddings_adapted.squeeze().unsqueeze(-1))
        anomaly_awareness_cls_patch = F.interpolate(anomaly_awareness_cls_patch.permute(0, 2, 1).view(-1, 1, 32, 32),
                                    size=self.img_size, mode='bilinear', align_corners=True)
        anomaly_awareness_cls_patch = torch.sigmoid(anomaly_awareness_cls_patch)
        return torch.cat([1 - anomaly_awareness_cls_patch, anomaly_awareness_cls_patch], dim=1)
    

    def global_anomaly_allignment(self, cls_embeddings_adapted, adapted_text_embeddings):
        return 100 * (torch.einsum('bd,bdk->bk', cls_embeddings_adapted , adapted_text_embeddings))
    
    def forward(self, text_embeddings, image_features):
        positive_text_adapted = self.positive_text_adapter(text_embeddings[:, 0, :])
        # positive_text_adapted = positive_text_adapted / (positive_text_adapted.norm(dim=-1, keepdim=True)+ 1e-8)
        negative_text_adapted = self.negative_text_adapter(text_embeddings[:, 1, :])
        # negative_text_adapted = negative_text_adapted / (negative_text_adapted.norm(dim=-1, keepdim=True)+ 1e-8)

        adapted_text_embeddings = torch.stack(
                [positive_text_adapted, negative_text_adapted], dim=-1
        ) 

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
            self.cross_model_contrastive_learning(adapted_text_embeddings, patch_embeddings_adapted[i])
            for i in range(self.number_of_vision_layers)], dim=1)

        anomaly_aware_calibration = torch.stack([
            self.anomaly_aware_calibration(patch_embeddings_adapted[i], cls_embeddings_adapted[i])
            for i in range(self.number_of_vision_layers)], dim=1)

        global_anomaly_alignment = torch.stack([
            self.global_anomaly_allignment(cls_embeddings_adapted[i], adapted_text_embeddings)
            for i in range(self.number_of_vision_layers)], dim=1)

        return cross_model_contrastive.mean(dim=1), anomaly_aware_calibration.mean(dim=1), global_anomaly_alignment.mean(dim=1)
