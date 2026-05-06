import os
from .base_dataset import BaseDataset

'''dataset source: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection'''

Br35h_CLS_NAMES = [
    'br35h',
]
Br35h_ROOT = "/media/data/ukhan/data/medical_cv/Br35h_anomaly_detection"

class Br35hDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(Br35hDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )

