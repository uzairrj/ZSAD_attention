import os
from .base_dataset import BaseDataset

'''dataset source: http://mv.cvc.uab.es/projects/colon-qa/cvccolondb'''
ColonDB_CLS_NAMES = [
    'endo',
]

ColonDB_ROOT = "/media/data/ukhan/data/medical_cv/Endo"

class EndoDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(EndoDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )


