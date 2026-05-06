from .constants import DatasetConstants
from .mvtec import MVTecDataset
from .visa import VisaDataset
from .btad import BTADDataset
from .mpdd import MPDDDataset
from .dtd import DTDDataset
from .ksdd import KSDDDataset
from .dagm import DAGMDataset
from .ksdd2 import KSDD2Dataset
from .isic import ISICDataset
from .tn3k import TN3KDataset
from .kvasir import KvasirDataset
from .clinicdb import ClinicDBDataset
from .colondb import ColonDBDataset
from .endo import EndoDataset
from .br35h import Br35hDataset
from .brain_mri import BrainMRIDataset
from .tn3k import TN3KDataset
from .headct import HEADCTDataset

base_dir_industrial = '/media/data/ukhan/data/computer_vision'
base_dir_medical = '/media/data/ukhan/data/medical_cv'


mvtec_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='MVTec')
visa_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='VisA')
btad_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='btad')
mpdd_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='MPDD')
dtd_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='DTD')
ksdd_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='KSDD')
ksdd2_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='KSDD2')
dagm_dataset_constants = DatasetConstants(base_path=base_dir_industrial, dataset_name='DAGM')

isic_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='ISIC')
tn3k_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='TN3K')
kvasir_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='Kvasir')
clinicdb_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='clinicdb')
colondb_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='colondb')
endo_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='endo')
br35h_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='br35h')
brain_mri_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='brainmri')
tn3k_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='TN3K')
headct_dataset_constants = DatasetConstants(base_path=base_dir_medical, dataset_name='headct')

dataset_dict = {
    'btad': (btad_dataset_constants.get_class_names(), BTADDataset, btad_dataset_constants.get_data_path()),
    'mpdd': (mpdd_dataset_constants.get_class_names(), MPDDDataset, mpdd_dataset_constants.get_data_path()),
    'mvtec': (mvtec_dataset_constants.get_class_names(), MVTecDataset, mvtec_dataset_constants.get_data_path()),
    'visa': (visa_dataset_constants.get_class_names(), VisaDataset, visa_dataset_constants.get_data_path()),
    'dtd': (dtd_dataset_constants.get_class_names(), DTDDataset, dtd_dataset_constants.get_data_path()),
    'ksdd': (ksdd_dataset_constants.get_class_names(), KSDDDataset, ksdd_dataset_constants.get_data_path()),
    'ksdd2': (ksdd2_dataset_constants.get_class_names(), KSDD2Dataset, ksdd2_dataset_constants.get_data_path()),
    'dagm': (dagm_dataset_constants.get_class_names(), DAGMDataset, dagm_dataset_constants.get_data_path()),

    'isic': (isic_dataset_constants.get_class_names(), ISICDataset, isic_dataset_constants.get_data_path()),
    'tn3k': (tn3k_dataset_constants.get_class_names(), TN3KDataset, tn3k_dataset_constants.get_data_path()),
    'kvasir': (kvasir_dataset_constants.get_class_names(), KvasirDataset, kvasir_dataset_constants.get_data_path()),
    'clinicdb': (clinicdb_dataset_constants.get_class_names(), ClinicDBDataset, clinicdb_dataset_constants.get_data_path()),
    'colondb': (colondb_dataset_constants.get_class_names(), ColonDBDataset, colondb_dataset_constants.get_data_path()),
    'endo': (endo_dataset_constants.get_class_names(), EndoDataset, endo_dataset_constants.get_data_path()),
    'br35h': (br35h_dataset_constants.get_class_names(), Br35hDataset, br35h_dataset_constants.get_data_path()),
    'brainmri': (brain_mri_dataset_constants.get_class_names(), BrainMRIDataset, brain_mri_dataset_constants.get_data_path()),
    'tn3k': (tn3k_dataset_constants.get_class_names(), TN3KDataset, tn3k_dataset_constants.get_data_path()),
    'headct': (headct_dataset_constants.get_class_names(), HEADCTDataset, headct_dataset_constants.get_data_path())
}

def get_data(dataset_name, transform, target_transform, training):
    dataset_cls_names, dataset_instance, dataset_root = dataset_dict[dataset_name]
    return dataset_instance(
                clsnames=dataset_cls_names,
                transform=transform,
                target_transform=target_transform,
                training=training,
                root=dataset_root
            )