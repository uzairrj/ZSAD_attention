import os
from .base_dataset import BaseDataset

'''dataset source: https://paperswithcode.com/dataset/cvc-clinicdb'''
ClinicDB_CLS_NAMES = [
    'Colon_clinicDB',
]

ClinicDB_ROOT = "/media/data/ukhan/data/medical_cv/ClinicDB"

class ClinicDBDataset(BaseDataset):
    def __init__(self, transform, target_transform, clsnames, root, training=True):
        super(ClinicDBDataset, self).__init__(
            clsnames=clsnames, transform=transform, target_transform=target_transform,
            root=root, training=training
        )
