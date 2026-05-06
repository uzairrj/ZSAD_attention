import os

from dataset_preprocess.colondb import ColonDB_ROOT

from dataset_preprocess.colondb import ColonDB_ROOT
from .base_dataset import BaseDataset

'''dataset source: http://mv.cvc.uab.es/projects/colon-qa/cvccolondb'''
ColonDB_CLS_NAMES = [
    'kvasie',
]

kvasir_ROOT = '/media/data/ukhan/data/medical_cv/Kvasir'

class KvasirDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(KvasirDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )


