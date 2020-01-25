'''
This is the main function running the Training, Validation, Testing process. Just run this file!
You can also change the model parameters here.
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
'''

# Deep Learning Modules
from torch.nn import *
import torch
import torch.optim as optim

# User Defined Modules
# from configs.serde import *
from Train_Test_Valid import Training
# from models.RNN import *
from data.data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback

#System Modules
from itertools import product
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def main_train():
    '''Main function for training.'''

    # '''Hyper-parameters'''
    # NUM_EPOCH = 15
    # LOSS_FUNCTION = CrossEntropyLoss
    # OPTIMIZER = optim.Adam
    # parameters = dict(lr = [.01], batch_size = [32])
    # param_values = [v for v in parameters.values()]
    #
    # '''Hyper-parameter testing'''
    # for lr, BATCH_SIZE in product(*param_values):
    #     # put the new experiment name here.
    #     params = create_experiment("newAdam_" + str(lr) +'_batch_size'+ str(BATCH_SIZE))
    #     cfg_path = params["cfg_path"]
    #
    #     '''Prepare data'''
    #     # Train Set
    #     full_train_dataset = data_provider(cfg_path=cfg_path, dataset_name='2014_b_train.txt', size=1000)
    #     VOCAB_SIZE = full_train_dataset.vocab_size
    #     train_size = int(0.8 * len(full_train_dataset))
    #     valid_size = len(full_train_dataset) - train_size
    #     train_dataset, test_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, valid_size])
    #
    #     # train_dataset = ConcatDataset([
    #     #     data_provider(dataset_name='2014_b_train.txt', size=10222, cfg_path=CFG_PATH),
    #     #     data_provider(dataset_name='2014_b_dev.txt', size=1000, cfg_path=CFG_PATH)])
    #     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
    #                                                drop_last=True, shuffle=True, num_workers=4)
    #     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
    #                                                drop_last=True, shuffle=True, num_workers=4)
    #     # Validation Set
    #     # test_dataset = ConcatDataset([
    #     #     data_provider(dataset_name='2015_b_test_gold.txt', size=500, cfg_path=cfg_path, mode=Mode.VALID, seed=5),
    #     #     data_provider(dataset_name='2014_b_test_gold.txt', size=500, cfg_path=cfg_path, mode=Mode.VALID, seed=5)])
    #     # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
    #     #                                           drop_last=True, shuffle=False, num_workers=4)
    #     '''Initialize trainer'''
    #     trainer = Training(cfg_path)
    #     '''Define model parameters'''
    #     optimiser_params = {'lr': lr}
    #     MODEL = GRUU(vocab_size=VOCAB_SIZE, batch_size=BATCH_SIZE)
    #     trainer.setup_model(model=MODEL, optimiser=OPTIMIZER,
    #                         optimiser_params=optimiser_params, loss_function=LOSS_FUNCTION)
    #     '''Execute Training'''
    #     trainer.execute_training(train_loader, test_loader=test_loader, num_epochs=NUM_EPOCH)



# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py

# set up your model


# set up loss (you can find preimplemented loss functions in t.nn) use the pos_weight parameter to ease convergence
# set up optimizer (see t.optim); 
# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer


# go, go, go... call fit on trainer
res = #TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')


def main_test():
    '''Main function for prediction'''
    pass



# def experiment_deleter():
#     '''Use below lines if you want to delete an experiment and reuse the same experiment name'''
#     parameters = dict(lr = [.01], batch_size = [32])
#     param_values = [v for v in parameters.values()]
#     for lr, BATCH_SIZE in product(*param_values):
#         delete_experiment("newAdam_" + str(lr) +'_batch_size'+ str(BATCH_SIZE))



if __name__ == '__main__':
    # experiment_deleter()
    main_train()
    # main_test()
