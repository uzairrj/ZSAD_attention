import torch
from transformers import AutoTokenizer, CLIPModel

class CLIPTextEncoder():

    def __init__(self, model_id="openai/clip-vit-large-patch14-336", device='cpu'):

        self.model_id = model_id
        self.device = device 

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

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
    
if __name__ == "__main__":
    encoder = CLIPTextEncoder()
    texts = ["a normal brain scan", "a damaged brain scan"]
    features = encoder(texts)
    print(features.shape)
