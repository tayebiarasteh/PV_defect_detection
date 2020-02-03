'''
This is the main function running the Training, Validation, Testing process.
Set the hyper-parameters and model parameters here. [data parameters from config file]

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

# Deep Learning Modules
from torch.nn import *
import torch
import torch.optim as optim

# User Defined Modules
from configs.serde import *
from Train_Test_Valid import *
from models.resnet import *
from data.data_handler import *

#System Modules
from itertools import product
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')




def main_train():
    '''Main function for training + validation'''

    '''Hyper-parameters'''
    NUM_EPOCH = 70
    LOSS_FUNCTION = BCEWithLogitsLoss
    OPTIMIZER = optim.Adam
    VALID_SPLIT_RATIO = 0.2
    parameters = dict(lr = [9e-4], batch_size = [32])
    param_values = [v for v in parameters.values()]

    '''Hyper-parameter testing'''
    for lr, BATCH_SIZE in product(*param_values):
        # put the new experiment name here.
        params = create_experiment("Adam_lr" + str(lr))
        cfg_path = params["cfg_path"]

        '''Prepare data'''
        train_dataset = get_train_dataset(cfg_path, valid_split_ratio=VALID_SPLIT_RATIO)
        valid_dataset = get_validation_dataset(cfg_path, valid_split_ratio=VALID_SPLIT_RATIO)
        pos_weight = train_dataset.pos_weight()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                                   drop_last=True, shuffle=True, num_workers=4)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                                                   drop_last=True, shuffle=False, num_workers=4)
        '''Initialize trainer'''
        trainer = Training(cfg_path, stopping_patience=5)
        '''Define model parameters'''
        optimiser_params = {'lr': lr}
        MODEL = ResNet()
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                            optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, pos_weight=pos_weight)
        '''Execute Training'''
        trainer.execute_training(train_loader, valid_loader=valid_loader, num_epochs=NUM_EPOCH, batch_size=BATCH_SIZE)




def main_test():
    '''Main function for prediction'''
    pass



def experiment_deleter():
    '''Use below lines if you want to delete an experiment and reuse the same experiment name'''
    parameters = dict(lr = [1e-2], batch_size = [1])
    param_values = [v for v in parameters.values()]
    for lr, BATCH_SIZE in product(*param_values):
        delete_experiment("new3Adam_lr" + str(lr))



if __name__ == '__main__':
    # experiment_deleter()
    main_train()
    # main_test()
