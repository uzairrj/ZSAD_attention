import os
from .base_dataset import BaseDataset

'''dataset source: https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection'''
DAGM_CLS_NAMES = [
    'Class1', 'Class2', 'Class3', 'Class4', 'Class5','Class6','Class7','Class8','Class9','Class10',
]

class DAGMDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(DAGMDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )
