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
import warnings
warnings.filterwarnings('ignore')




def main_train():
    '''Main function for training + validation'''

    # if we are resuming training on a model
    RESUME = False

    # Hyper-parameters
    NUM_EPOCH = 200
    LOSS_FUNCTION = BCEWithLogitsLoss
    OPTIMIZER = optim.Adam
    VALID_SPLIT_RATIO = 0.2
    lr = 5e-5
    BATCH_SIZE = 32


    if RESUME == True:
        EXPERIMENT_NAME = 'Adam_lr5e-05'
        params = open_experiment(EXPERIMENT_NAME)
    else:
        # put the new experiment name here.
        params = create_experiment("Adam_lr" + str(lr))
    cfg_path = params["cfg_path"]

    # Prepare data
    train_dataset = get_train_dataset(cfg_path, valid_split_ratio=VALID_SPLIT_RATIO)
    pos_weight = train_dataset.pos_weight()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                               drop_last=False, shuffle=True, num_workers=4)
    if VALID_SPLIT_RATIO != 0:
        valid_dataset = get_validation_dataset(cfg_path, valid_split_ratio=VALID_SPLIT_RATIO)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE,
                                                   drop_last=True, shuffle=False, num_workers=4)
    else:
        valid_loader = None

    # Initialize trainer
    trainer = Training(cfg_path, stopping_patience=10, num_epochs=NUM_EPOCH, RESUME=RESUME)
    # Define model parameters
    optimiser_params = {'lr': lr, 'weight_decay': 1e-5}
    MODEL = ResNet()

    if RESUME == True:
        trainer.load_checkpoint(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, pos_weight=pos_weight)
    else:
        trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
                        optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION, pos_weight=pos_weight)
    # Execute Training
    trainer.execute_training(train_loader, valid_loader=valid_loader, batch_size=BATCH_SIZE)





def main_test():
    '''Main function for prediction'''
    EXPERIMENT_NAME = 'Adam_lr5e-05'
    params = open_experiment(EXPERIMENT_NAME)
    cfg_path = params['cfg_path']

    '''Initialize predictor'''
    predictor = Prediction(cfg_path)
    predictor.setup_model(ResNet)
    # export onnx file
    predictor.save_onnx(params['network_output_path'] + '/' + 'checkpoint.onnx')


def experiment_deleter():
    '''Use below lines if you want to delete an experiment and reuse the same experiment name'''
    parameters = dict(lr = [5e-5], batch_size = [1])
    param_values = [v for v in parameters.values()]
    for lr, BATCH_SIZE in product(*param_values):
        delete_experiment("Adam_lr" + str(lr))



if __name__ == '__main__':
    # experiment_deleter()
    # main_train()
    main_test()
