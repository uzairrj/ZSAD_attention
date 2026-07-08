import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel, AutoProcessor

class CLIPTextEncoder():

    def __init__(self, model_id="openai/clip-vit-large-patch14-336", layers=[-1], device='cpu', image_size=None):

        self.model_id = model_id
        self.device = device 

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id)

        if image_size is not None and image_size != self.model.config.vision_config.image_size:
            orig_image_size = self.model.config.vision_config.image_size
            patch_size = self.model.config.vision_config.patch_size
            
            orig_grid_size = (orig_image_size // patch_size, orig_image_size // patch_size)
            new_grid_size = (image_size // patch_size, image_size // patch_size)
            
            old_pos_embed = self.model.vision_model.embeddings.position_embedding.weight.data
            
            extra_tokens = 1
            pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
            
            pos_emb_img = pos_emb_img.reshape(1, orig_grid_size[0], orig_grid_size[1], -1).permute(0, 3, 1, 2)
            pos_emb_img = F.interpolate(
                pos_emb_img,
                size=new_grid_size,
                mode='bicubic',
                antialias=True,
                align_corners=False,
            )
            pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(new_grid_size[0] * new_grid_size[1], -1)
            new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
            
            self.model.config.vision_config.image_size = image_size
            self.model.vision_model.config.image_size = image_size
            self.model.vision_model.embeddings.image_size = image_size
            
            num_patches = new_grid_size[0] * new_grid_size[1]
            num_positions = num_patches + 1
            self.model.vision_model.embeddings.num_patches = num_patches
            self.model.vision_model.embeddings.num_positions = num_positions
            
            hidden_size = self.model.config.vision_config.hidden_size
            self.model.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, hidden_size)
            self.model.vision_model.embeddings.position_embedding.weight.data.copy_(new_pos_embed)
            self.model.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if image_size is not None:
            if hasattr(self.processor, "image_processor"):
                if hasattr(self.processor.image_processor, "size") and isinstance(self.processor.image_processor.size, dict):
                    if "shortest_edge" in self.processor.image_processor.size:
                        self.processor.image_processor.size = {"shortest_edge": image_size}
                    else:
                        self.processor.image_processor.size = {"height": image_size, "width": image_size}
                if hasattr(self.processor.image_processor, "crop_size") and isinstance(self.processor.image_processor.crop_size, dict):
                    self.processor.image_processor.crop_size = {"height": image_size, "width": image_size}

        self.layers = layers

    def __call__(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs).pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, images):


        with torch.inference_mode():
            outputs = self.model.vision_model(
                pixel_values=images.to(self.device),
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states

        selected = [hidden_states[layer] for layer in self.layers]

        cls = torch.stack([h[:, 0, :] for h in selected], dim=1)
        patches = torch.stack([h[:, 1:, :] for h in selected], dim=1)
        return cls, patches
    
        
    
if __name__ == "__main__":
    encoder = CLIPTextEncoder(image_size=518)
    texts = ["a normal brain scan", "a damaged brain scan"]
    features = encoder(texts)
    print("Text features shape:", features.shape)
    
    # Test with random image
    import torch
    dummy_image = torch.randn(1, 3, 518, 518)
    cls, patches = encoder.get_image_features(dummy_image)
    print("Image features cls shape:", cls.shape)
    print("Image features patches shape:", patches.shape)

