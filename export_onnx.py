import torch
from configs.serde import *
from Train_Test_Valid import *
import sys
import torchvision as tv

# Deep Learning Modules
from torch.nn import *
import torch
import torch.optim as optim

# User Defined Modules
from configs.serde import *
from Train_Test_Valid import *
from models.resnet import *
from data.data_handler import *

# epoch = int(sys.argv[1])
epoch = 5


EXPERIMENT_NAME = 'new2Adam_lr0.01'
params = open_experiment(EXPERIMENT_NAME)
cfg_path = params['cfg_path']

'''Initialize predictor'''
predictor = Prediction(cfg_path)
predictor.setup_model(ResNet, epoch)

# trainer.restore_checkpoint(epoch)
predictor.save_onnx(params['network_output_path'] + '/' + 'checkpoint_{:03d}.onnx'.format(epoch))
