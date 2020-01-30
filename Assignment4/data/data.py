"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""


from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from Train_Test_Valid import Mode
from configs.serde import read_config

import pdb

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15
# HEIGHT=sc['height']
# WIDTH=sc['width']



class ChallengeDataset(Dataset):
    def __init__(self, cfg_path, split_ratio=0.8, composer=transforms.Compose(transforms.ToTensor()), mode=Mode.TRAIN, seed=1):
        '''
        Args:
            cfg_path (string):
                Config file path of the experiment
            split_ratio (float):
                train-valid splitting
            mode (enumeration Mode):
                Nature of operation to be done with the data.
                Possible inputs are Mode.PREDICTION, Mode.TRAIN, Mode.VALID, Mode.TEST
                Default value: Mode.TRAIN
        '''
        params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.mode = mode
        self.seed = seed
        self.split_ratio = split_ratio

        self.composer = composer
        self.input_list = pd.read_csv(params['input_csv_path'], delimiter=';')


    def __len__(self):
        '''Returns length of the dataset'''
        return len(self.input_list)

    def __getitem__(self, idx):
        '''
        transforms also here
        :return:
            label: e.g. tensor([1, 0]), first element corresponds to "crack" and second to "inactive"
        '''
        label_temp = np.asarray(self.input_list.loc[self.input_list['filename'] == self.input_list['filename'][idx]])[0, 2:]
        label = np.zeros((2), dtype=int)
        label[0] = label_temp[0]
        label[1] = label_temp[1]

        # Reads images using files name available in the list
        image = imread(self.input_list['filename'][idx])
        # image = resize(image, (HEIGHT, WIDTH))
        image = gray2rgb(image)

        # Conversion to ubyte value range (0...255) is done here,
        # because network needs to be trained and needs to predict using the same datatype.
        image = img_as_ubyte(image)

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label


    def pos_weight(self):
        '''Calculates a weight for positive examples for each class and returns it as a tensor'''
        w_crack = torch.tensor((1 - self.input_list['crack'].sum()) / (self.input_list['crack'].sum() + epsilon))
        w_inactive = torch.tensor((1 - self.input_list['inactive'].sum()) / (self.input_list['inactive'].sum() + epsilon))
        return w_crack, w_inactive



# def get_train_dataset():
#     trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
#     return ChallengeDataset(csv_path='./train.csv', split_ratio=0.8, composer=trans)
#
# # this needs to return a dataset *without* data augmentation!
# def get_validation_dataset():
#     trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
#     return ChallengeDataset(csv_path='./train.csv', split_ratio=0.2, composer=trans)




if __name__ == '__main__':
    CONFIG_PATH = '/home/soroosh/Documents/Repositories/deep_learning_challenge/Assignment4/configs/config.json'
    dataset = ChallengeDataset(cfg_path=CONFIG_PATH, split_ratio=0.8, mode=Mode.TRAIN, seed=1)
    # pdb.set_trace()
    # a=5