# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os.path as osp
import os
import gdown
import pandas as pd

data_dir = '../../data'

def get_gtr_data():
    """Return grasp type recognition pretrained model path.

    Returns
    -------
    dst_path : str
        pretrained model path.
    """
    dst_path = osp.join(data_dir, 'grasp-type-recognition-resnet101.pth')
    if not osp.exists(dst_path):
        raise (ValueError, "Dataset is not found. Please download it using github LFS")
    return dst_path


def get_gtr_dataset():
    dst_path = osp.join(data_dir, 'grasp_type_recognition_dataset.tar.gz')
    if not osp.exists(dst_path):
        raise (ValueError, "Dataset is not found. Please download it using github LFS")
    return osp.join(data_dir, 'grasp_type_recognition_dataset')


def get_uniformal_affordance():
    dst_path = osp.join(data_dir, 'uniformal_affordance.csv')
    if not osp.exists(dst_path):
        raise (ValueError, "Dataset is not found. Please download it using github LFS")
    return pd.read_csv(dst_path, index_col=0)


def get_varied_affordance():
    dst_path = osp.join(data_dir, 'varied_affordance.csv')
    if not osp.exists(dst_path):
        raise (ValueError, "Dataset is not found. Please download it using github LFS")
    return pd.read_csv(dst_path, index_col=0)
