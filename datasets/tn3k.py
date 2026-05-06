import os
from .base_dataset import BaseDataset

'''dataset source: https://ieeexplore.ieee.org/document/9434087/references#references'''
TN3K_CLS_NAMES = [
    'tn3k',
]

class TN3KDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(TN3KDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )


