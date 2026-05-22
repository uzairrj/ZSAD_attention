import torchvision.transforms as transforms
from PIL import Image

DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGE_STD = (0.229, 0.224, 0.225)

CLIP_MEAN= (
    0.48145466,
    0.4578275,
    0.40821073
)
CLIP_STD= (
    0.26862954,
    0.26130258,
    0.27577711
)

def get_transforms(img_size):
    transform_img = transforms.Compose( [
                transforms.Resize((img_size, img_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ])
    
    transform_mask = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
    
    return transform_img, transform_mask
