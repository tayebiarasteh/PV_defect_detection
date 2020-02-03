# Deep Learning Modules
import torch
from torch.nn import *
import torch.optim as optim

# User Defined Modules
from configs.serde import *
from Train_Test_Valid import *
from models.resnet import *
from data.data_handler import *


EXPERIMENT_NAME = 'Adam_lr0.0009'
params = open_experiment(EXPERIMENT_NAME)
cfg_path = params['cfg_path']

'''Initialize predictor'''
predictor = Prediction(cfg_path)
predictor.setup_model(ResNet)

predictor.save_onnx(params['network_output_path'] + '/' + 'checkpoint.onnx')
