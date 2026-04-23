from pyexpat import model

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOImageEncoder():
    def __init__(self, model_id, layers, device="cuda"):
        self.model_id = model_id
        self.layers = layers
        self.device = device

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def __call__(self, images):
        # inputs = self.processor(images=images, return_tensors="pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        inputs = {"pixel_values": images.to(self.device)}

        with torch.no_grad():
            hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states

        selected = [hidden_states[layer] for layer in self.layers]
        cls = torch.stack([h[:, 0, :] for h in selected], dim=1)
        patches = torch.stack([h[:, 5:, :] for h in selected], dim=1)

        return cls, patches