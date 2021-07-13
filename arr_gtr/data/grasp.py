# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os.path as osp

import gdown
import pandas as pd

download_dir = osp.expanduser('~/.task_model_recognition')
data_dir = osp.join(download_dir, 'data', 'grasp')

def get_gtr_data():
    """Return grasp type recognition pretrained model path.

    Returns
    -------
    dst_path : str
        pretrained model path.
    """
    dst_path = osp.join(data_dir, 'grasp-type-recognition-resnet101.pth')
    url = 'https://drive.google.com/uc?id=1tnj1iQzV33Fv5RlgNazZr0cZxEqZmAI5'
    if not osp.exists(dst_path):
        gdown.cached_download(
            url=url,
            path=dst_path,
            md5='74190379b285b4053f2bc70baa6f9999',
            quiet=True,
        )
    return dst_path


def get_gtr_dataset():
    dst_path = osp.join(data_dir, 'grasp_type_recognition_dataset.tar.gz')
    url = 'https://drive.google.com/uc?id=1yDdF6MWxW6UHuNHztVBcX4G-GW1YsocI'
    if not osp.exists(dst_path):
        gdown.cached_download(
            url=url,
            path=dst_path,
            md5='ee31073e2d125ef7787830a14842ec1c',
            postprocess=gdown.extractall,
            quiet=True,
        )
    return osp.join(data_dir, 'grasp_type_recognition_dataset')


def get_uniformal_affordance():
    dst_path = osp.join(data_dir, 'uniformal_affordance.csv')
    url = 'https://drive.google.com/uc?id=1b7CfSQuOjvqTgv29igD7mn1oyhg_0nAC'
    if not osp.exists(dst_path):
        gdown.cached_download(
            url=url,
            path=dst_path,
            md5='d7217f1cfd8f40706cdf23bf016a6bff',
            quiet=True,
        )
    return pd.read_csv(dst_path, index_col=0)


def get_varied_affordance():
    dst_path = osp.join(data_dir, 'varied_affordance.csv')
    url = 'https://drive.google.com/uc?id=1r_M4XxUl1tXQ6RqO4QKS8eGpJMh8DWaV'
    if not osp.exists(dst_path):
        gdown.cached_download(
            url=url,
            path=dst_path,
            md5='a3921c165afcdb8bffe7bad7630da8b6',
            quiet=True,
        )
    return pd.read_csv(dst_path, index_col=0)
