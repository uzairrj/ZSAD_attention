from torch.nn import Linear, Module, ModuleList, Parameter, functional as F
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


        self.text_logit_scale = Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self.cross_model_contrastive_learning_attention = ModuleList([CrossAttention(args.out_dim, 1, 0.2)
                                                                      for _ in range(self.number_of_vision_layers)])

        self.cross_model_contrastive_head = ModuleList([Linear(args.out_dim, 1)
                                                        for _ in range(self.number_of_vision_layers)])

        self.layer_logits = Parameter(torch.zeros(self.number_of_vision_layers))
        self.cls_layer_logits = Parameter(torch.zeros(self.number_of_vision_layers))
        self.global_patch_layer_logits = Parameter(torch.zeros(self.number_of_vision_layers))
        self.global_fusion_logits = Parameter(torch.zeros(2))
        self.global_topk_ratio = getattr(args, "global_topk_ratio", 0.01)
        
        self.img_size = args.img_size

    def normalize_features(self, features):
        return features / (features.norm(dim=-1, keepdim=True) + 1e-8)

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
        cross_modal_patch_logits = self.cross_model_contrastive_head[layer_idx](patch_features)
        anomaly_map_cross_modal = F.interpolate(self.patch_logits_to_map(cross_modal_patch_logits),
                                    size=self.img_size, mode='bilinear', align_corners=True)
        return anomaly_map_cross_modal, cross_modal_patch_logits

    def text_contrast_logits(self, image_embeddings, normal_text_adapted, abnormal_text_adapted):
        image_embeddings = self.normalize_features(image_embeddings)
        logit_scale = self.text_logit_scale.exp().clamp(max=100)
        normal_logits = torch.einsum(
            'bd,bnd->bn',
            image_embeddings,
            normal_text_adapted,
        )
        abnormal_logits = torch.einsum(
            'bd,bnd->bn',
            image_embeddings,
            abnormal_text_adapted,
        )
        normal_score = normal_logits.mean(dim=1, keepdim=True)
        abnormal_score = abnormal_logits.mean(dim=1, keepdim=True)
        return logit_scale * (abnormal_score - normal_score)

    def cls_text_contrast_logits(self, cls_embeddings_adapted, normal_text_adapted, abnormal_text_adapted):
        layer_logits = torch.stack([
            self.text_contrast_logits(cls_embeddings_adapted[i], normal_text_adapted, abnormal_text_adapted)
            for i in range(self.number_of_vision_layers)
        ], dim=1)
        layer_weights = torch.softmax(self.cls_layer_logits, dim=0)
        return (layer_logits * layer_weights.view(1, -1, 1)).sum(dim=1)

    def topk_cross_modal_logits(self, cross_modal_patch_logits):
        layer_logits = []

        for patch_logits in cross_modal_patch_logits:
            patch_logits = patch_logits.squeeze(-1)
            patch_count = patch_logits.shape[1]
            topk = max(1, int(patch_count * self.global_topk_ratio))
            layer_logits.append(patch_logits.topk(topk, dim=1).values.mean(dim=1, keepdim=True).detach())

        layer_logits = torch.stack(layer_logits, dim=1)
        layer_weights = torch.softmax(self.global_patch_layer_logits, dim=0)
        return (layer_logits * layer_weights.view(1, -1, 1)).sum(dim=1)

    def global_text_contrast_logits(
        self,
        cls_embeddings_adapted,
        cross_modal_patch_logits,
        normal_text_adapted,
        abnormal_text_adapted,
    ):
        # Stop the global CLS/text loss from training text adapters,
        # while keeping gradients into the CLS adapters.
        cls_normal_text_adapted = normal_text_adapted.detach()
        cls_abnormal_text_adapted = abnormal_text_adapted.detach()
        cls_logits = self.cls_text_contrast_logits(
            cls_embeddings_adapted,
            cls_normal_text_adapted,
            cls_abnormal_text_adapted,
        )
        patch_logits = self.topk_cross_modal_logits(cross_modal_patch_logits)
        fusion_weights = torch.softmax(self.global_fusion_logits, dim=0)
        return 1 * cls_logits + 1 * patch_logits
    
    def forward(self, text_embeddings, image_features):
        normal_text_embeddings, abnormal_text_embeddings = self.unpack_text_embeddings(text_embeddings)
        positive_text_adapted = self.positive_text_adapter(normal_text_embeddings)
        positive_text_adapted = self.normalize_features(positive_text_adapted)
        negative_text_adapted = self.negative_text_adapter(abnormal_text_embeddings)
        negative_text_adapted = self.normalize_features(negative_text_adapted)
        adapted_text_embeddings = self.merge_text_embeddings(positive_text_adapted, negative_text_adapted)

        cls_embeddings_adapted = []
        for i in range(self.number_of_vision_layers):
            cls_embedding_adapted = self.cls_adapter[i](image_features[0][:, i, :])
            cls_embedding_adapted = self.normalize_features(cls_embedding_adapted)
            cls_embeddings_adapted.append(cls_embedding_adapted)

        patch_embeddings_adapted = []
        for i in range(self.number_of_vision_layers):
            patch_embedding_adapted = self.patch_adapter[i](image_features[1][:,i,:,:])
            patch_embedding_adapted = self.normalize_features(patch_embedding_adapted)
            patch_embeddings_adapted.append(patch_embedding_adapted)

        cross_model_contrastive = []
        cross_modal_patch_logits = []
        for i in range(self.number_of_vision_layers):
            anomaly_map_cross_modal, patch_logits = self.cross_model_contrastive_learning(
                adapted_text_embeddings,
                patch_embeddings_adapted[i],
                i,
            )
            cross_model_contrastive.append(anomaly_map_cross_modal)
            cross_modal_patch_logits.append(patch_logits)
        cross_model_contrastive = torch.stack(cross_model_contrastive, dim=1)

        global_anomaly_alignment = self.global_text_contrast_logits(
            cls_embeddings_adapted,
            cross_modal_patch_logits,
            positive_text_adapted,
            negative_text_adapted,
        )

        weights = torch.softmax(self.layer_logits, dim=0)
        final_map = (cross_model_contrastive * weights.view(1, -1, 1, 1, 1)).sum(dim=1)

        return final_map, global_anomaly_alignment
