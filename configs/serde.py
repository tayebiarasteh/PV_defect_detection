"""
functions for writing/reading data to/from disk
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

import json
import numpy as np
import os
import warnings
import shutil


CONFIG_PATH = '/home/soroosh/Documents/Repositories/deep_learning_challenge/configs/config.json'


def read_config(cfg_path):
    with open(cfg_path, 'r') as f:
        params = json.load(f)
    return params


def write_config(params, cfg_path, sort_keys=False):
    with open(cfg_path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=sort_keys)


def create_experiment(experiment_name):
    params = read_config(CONFIG_PATH)
    params['experiment_name'] = experiment_name
    create_experiment_folders(params)
    cfg_file_name = params['experiment_name'] + '_config.json'
    cfg_path = os.path.join(params['network_output_path'], cfg_file_name)
    params['cfg_path'] = cfg_path
    write_config(params, cfg_path)
    return params


def create_experiment_folders(params):
    try:
        path_keynames = ["network_output_path", "output_data_path", "tb_logs_path"]
        for key in path_keynames:
            params[key] = os.path.join(params[key], params['experiment_name'])
            os.makedirs(params[key])
    except:
        raise Exception("Experiment already exist. Please try a different experiment name")


def open_experiment(experiment_name):
    '''Open Existing Experiments'''
    default_params = read_config(CONFIG_PATH)
    cfg_file_name = experiment_name + '_config.json'
    cfg_path = os.path.join(default_params['network_output_path'], experiment_name, cfg_file_name)
    params = read_config(cfg_path)
    return params


def delete_experiment(experiment_name):
    '''Delete Existing Experiment folder'''
    default_params = read_config(CONFIG_PATH)
    cfg_file_name = experiment_name + '_config.json'
    cfg_path = os.path.join(default_params['network_output_path'], experiment_name, cfg_file_name)
    params = read_config(cfg_path)
    path_keynames = ["network_output_path", "output_data_path", "tb_logs_path"]
    for key in path_keynames:
        shutil.rmtree(params[key])


def create_retrain_experiment(experiment_name, source_pth_file_path):
    params = create_experiment(experiment_name)
    params['Network']['retrain'] = True
    destination_pth_file_path = os.path.join(params['network_output_path'], 'pretrained_model.pth')
    params['Network']['pretrain_model_path'] = destination_pth_file_path
    shutil.copy(source_pth_file_path, destination_pth_file_path)
    write_config(params, params['cfg_path'])
    return params