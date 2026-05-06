import os
from .base_dataset import BaseDataset

ISIC_CLS_NAMES = [
    'isic',
]

class ISICDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames,  root, training=True):
        super(ISICDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )


