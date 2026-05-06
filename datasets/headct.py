import os
from .base_dataset import BaseDataset

'''dataset source: https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage'''
HEADCT_CLS_NAMES = [
    'headct',
]

class HEADCTDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(HEADCTDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )


