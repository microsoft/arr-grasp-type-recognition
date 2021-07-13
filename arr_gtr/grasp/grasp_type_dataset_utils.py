# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import division

import os
import os.path as osp

import numpy as np
import pandas as pd

from arr_gtr.grasp.grasp_type_utils import grasp_types


object_names = (
    'BleachCleanser',
    'GelatinBox',
    'PottedMeatCan',
    'MetalBowl',
    'Spatula',
    'Skillet',
    'Mug',
    'Plate',
    'Fork',
    'AbrasiveSponge',
    'CrackerBox',
    'Apple',
    'Pitcher',
    'ChipsCan',
    'Pear',
    'Banana',
    'Knife',
    'Peach',
    'Spoon',
)


def calculate_pg(dataset_path):
    """Calculate p(g)

    """
    pg = {}
    for grasp in grasp_types:
        cnt = 0
        for obj_name in os.listdir(dataset_path):
            for img in os.listdir(osp.join(dataset_path, obj_name)):
                if grasp in img:
                    cnt += 1
        pg[grasp] = cnt
    cnt = 0
    for key, item in pg.items():
        cnt += item
    for key, item in pg.items():
        pg[key] /= cnt
    return pg


def calculate_uniformal_affordance(directory_path):
    """Calculate uniformal affordance p(g|o)

    """
    uniformal_affordance = {}
    for obj in object_names:
        uniformal_affordance[obj] = {}
        for grasp in grasp_types:
            uniformal_affordance[obj][grasp] = 0
    for grasp in grasp_types:
        for obj_name in object_names:
            for img in os.listdir(osp.join(directory_path, obj_name)):
                if grasp in img:
                    if uniformal_affordance[obj_name][grasp] == 0:
                        uniformal_affordance[obj_name][grasp] = 1
    for key, item in uniformal_affordance.items():
        cnt = 0
        for k, ii in item.items():
            cnt += ii
        for k, ii in item.items():
            uniformal_affordance[key][k] /= cnt
    return uniformal_affordance


def calculate_varied_affordance(directory_path):
    """Calculate varied affordance p(g|o)

    """
    varied_affordance = {}
    for obj_name in object_names:
        varied_affordance[obj_name] = {}
        for grasp in grasp_types:
            varied_affordance[obj_name][grasp] = 0
    for grasp in grasp_types:
        for obj_name in object_names:
            for img in os.listdir(osp.join(directory_path, obj_name)):
                if grasp in img:
                    varied_affordance[obj_name][grasp] += 1 / 100.0
    return varied_affordance


def save_affordance(affordance_dict, output_path):
    object_names = list(affordance_dict.keys())

    dists = np.zeros((len(grasp_types), len(object_names)),
                     dtype=np.float32)
    for i, grasp_type in enumerate(grasp_types):
        for j, obj_name in enumerate(object_names):
            dists[i][j] = affordance_dict[obj_name][grasp_type]
    df = pd.DataFrame(np.array(dists), index=grasp_types,
                      columns=object_names)
    df.to_csv(output_path)
