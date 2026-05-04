import os
from .base_dataset import BaseDataset

'''dataset source: https://data.vicos.si/datasets/KSDD/KolektorSDD.zip'''
SDD_CLS_NAMES = [
    'SDD2',
]

class KSDD2Dataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(KSDD2Dataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )

