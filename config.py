import os
import yaml
import random
import pickle
from itertools import product
from datetime import datetime
from collections import OrderedDict

import numpy as np

from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F

# # Checkpoint code
import os
import zipfile
from shutil import copyfile


class Config():
    
    def __init__(self, config_name, slurm_job_id, config_dir='configs', load_from_exp=False):
        self.config_name = config_name
        self.slurm_job_id = slurm_job_id
        self.config_dir = config_dir

        if load_from_exp:
            self.timestamp_path = config_name
            self.figures_path = os.path.join(self.timestamp_path, "figures")
            self.models_path = os.path.join(self.timestamp_path, "models")
            self.checkpoints_path = os.path.join(self.timestamp_path, "checkpoints")
            self.results_path = os.path.join(self.timestamp_path, "results")
            config_load_path = os.path.join(self.checkpoints_path, 'config.yaml')
        else:
            config_load_path = os.path.join(config_dir, config_name)
            
        with open(config_load_path, 'r') as stream:
            try:
                self.config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc, flush=True)

        self.config_dict['data']['root'] = '../data' if not 'DATA_ROOT' in os.environ else os.environ['DATA_ROOT']
        self.config_dict['exp']['root'] = '../exps' # if not 'EXP_ROOT' in os.environ else os.environ['EXP_ROOT']

        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.config_dict['device'] = 'cpu'

        if not load_from_exp:
            self._init_paths()
            print(f'Experiment folders created in {self.timestamp_path}', flush=True)
        if not load_from_exp:
            self.checkpoint()
            print(f'Checkpoint created in {self.checkpoints_path}', flush=True)

        self.training_complete = False

    def _init_paths(self):
        self.benchmark_path = os.path.join(self['exp']['root'], self.config_dict['exp']['benchmark'])
        self.exp_path = os.path.join(self.benchmark_path, self['exp']['name'])
        timestamp = datetime.now().strftime("%d-%B-%Y-%H:%M:%S:%f")
        self.timestamp_path = os.path.join(self.exp_path, timestamp)
        self.figures_path = os.path.join(self.timestamp_path, "figures")
        self.models_path = os.path.join(self.timestamp_path, "models")
        self.checkpoints_path = os.path.join(self.timestamp_path, "checkpoints")
        self.results_path = os.path.join(self.timestamp_path, "results")

        if not os.path.isdir(self.benchmark_path):
            os.mkdir(self.benchmark_path)
        if not os.path.isdir(self.exp_path):
            os.mkdir(self.exp_path)
        if not os.path.isdir(self.timestamp_path):
            os.mkdir(self.timestamp_path)
            os.mkdir(self.figures_path)
            os.mkdir(self.models_path)
            os.mkdir(self.checkpoints_path)
            os.mkdir(self.results_path)

    def checkpoint(self):
        dst = os.path.join(self.checkpoints_path, "code-chkpt.zip")
        # Create a ZipFile Object
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zipObj:
           # Add multiple files to the zip
           zipObj.write('main.py')
           zipObj.write('config.py')
           zipObj.write('trainers.py')
           zipObj.write('models.py')
           zipObj.write('data.py')
           zipObj.write('utils.py')
           zipObj.write('finetuner.py')

        src = os.path.join('configs', self.config_name)
        dst = os.path.join(self.checkpoints_path, "config.yaml")
        copyfile(src, dst)

    def copy_log_to_exp_dir(self):
        log_file_name = f'slurm-{self.slurm_job_id}.out'
        src = os.path.join('scripts', 'logs', log_file_name)
        dst = os.path.join(self.timestamp_path, log_file_name)
        copyfile(src, dst)
    
    def print_config(self):
        pprint(vars(self))

    def __getitem__(self, key):
        return self.config_dict[key]
