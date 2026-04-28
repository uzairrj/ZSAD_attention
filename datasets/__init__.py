from .constants import DatasetConstants
from .mvtec import MVTecDataset
from .visa import VisaDataset
# from .mpdd import MPDD_CLS_NAMES, MPDDDataset, MPDD_ROOT
from .btad import BTADDataset
from .mpdd import MPDDDataset
# from .sdd import SDD_CLS_NAMES, SDDDataset, SDD_ROOT
# from .dagm import DAGM_CLS_NAMES, DAGMDataset, DAGM_ROOT
# from .dtd import DTD_CLS_NAMES,DTDDataset,DTD_ROOT
# from .isic import ISIC_CLS_NAMES,ISICDataset,ISIC_ROOT
# from .colondb import ColonDB_CLS_NAMES, ColonDBDataset, ColonDB_ROOT
# from .clinicdb import ClinicDB_CLS_NAMES, ClinicDBDataset, ClinicDB_ROOT
# from .tn3k import TN3K_CLS_NAMES, TN3KDataset, TN3K_ROOT
# from .headct import HEADCT_CLS_NAMES,HEADCTDataset,HEADCT_ROOT
# from .brain_mri import BrainMRI_CLS_NAMES,BrainMRIDataset,BrainMRI_ROOT
# from .br35h import Br35h_CLS_NAMES,Br35hDataset,Br35h_ROOT
# from torch.utils.data import ConcatDataset

base_dir = '/media/data/ukhan/data/computer_vision'

mvtec_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='MVTec')
visa_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='VisA')
btad_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='btad')
mpdd_dataset_constants = DatasetConstants(base_path=base_dir, dataset_name='MPDD')

dataset_dict = {
    # 'br35h': (Br35h_CLS_NAMES, Br35hDataset, Br35h_ROOT),
    # 'brain_mri': (BrainMRI_CLS_NAMES, BrainMRIDataset, BrainMRI_ROOT),
    'btad': (btad_dataset_constants.get_class_names(), BTADDataset, btad_dataset_constants.get_data_path()),
    # 'clinicdb': (ClinicDB_CLS_NAMES, ClinicDBDataset, ClinicDB_ROOT),
    # 'colondb': (ColonDB_CLS_NAMES, ColonDBDataset, ColonDB_ROOT),
    # 'dagm': (DAGM_CLS_NAMES, DAGMDataset, DAGM_ROOT),
    # 'dtd': (DTD_CLS_NAMES, DTDDataset, DTD_ROOT),
    # 'headct': (HEADCT_CLS_NAMES, HEADCTDataset, HEADCT_ROOT),
    # 'isic': (ISIC_CLS_NAMES, ISICDataset, ISIC_ROOT),
    'mpdd': (mpdd_dataset_constants.get_class_names(), MPDDDataset, mpdd_dataset_constants.get_data_path()),
    # 'sdd': (SDD_CLS_NAMES, SDDDataset, SDD_ROOT),
    # 'tn3k': (TN3K_CLS_NAMES, TN3KDataset, TN3K_ROOT),
    'mvtec': (mvtec_dataset_constants.get_class_names(), MVTecDataset, mvtec_dataset_constants.get_data_path()),
    'visa': (visa_dataset_constants.get_class_names(), VisaDataset, visa_dataset_constants.get_data_path()),
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