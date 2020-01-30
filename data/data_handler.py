"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from Train_Test_Valid import Mode
from configs.serde import read_config
import os.path
import pdb

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15



class ChallengeDataset(Dataset):
    def __init__(self, cfg_path, valid_split_ratio=0.2, transforms=transforms.Compose(transforms.ToTensor()), mode=Mode.TRAIN, seed=42):
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
        self.transforms = transforms
        self.input_list = pd.read_csv(params['input_csv_path'], delimiter=';')
        self.train_list, self.valid_list = train_test_split(self.input_list, test_size=valid_split_ratio, random_state=seed)


    def __len__(self):
        '''Returns length of the dataset'''
        if self.mode==Mode.TRAIN:
            return len(self.train_list)
        elif self.mode==Mode.VALID:
            return len(self.valid_list)

    def __getitem__(self, idx):
        '''
        :return:
            label: e.g. tensor([1, 0]), first element corresponds to "crack" and second to "inactive"
        '''
        if self.mode==Mode.TRAIN:
            output_list = self.train_list
        elif self.mode==Mode.VALID:
            output_list = self.valid_list

        label_temp = np.asarray(output_list.loc[output_list['filename'] == output_list['filename'][idx]])[0, 2:]
        label = np.zeros((2), dtype=int)
        label[0] = label_temp[0]
        label[1] = label_temp[1]

        # Reads images using files name available in the list
        image = imread(os.path.join('data', output_list['filename'][idx]))
        image = gray2rgb(image)
        image = self.transforms(image)
        label = torch.from_numpy(label)
        return image, label


    def pos_weight(self):
        '''Calculates a weight for positive examples for each class and returns it as a tensor'''
        w_crack = torch.tensor((1 - self.input_list['crack'].sum()) / (self.input_list['crack'].sum() + epsilon))
        w_inactive = torch.tensor((1 - self.input_list['inactive'].sum()) / (self.input_list['inactive'].sum() + epsilon))
        return w_crack, w_inactive



def get_train_dataset(cfg_path, valid_split_ratio):
    # since all the images are 300 * 300, we don't need resizing
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
    return ChallengeDataset(cfg_path=cfg_path, valid_split_ratio=valid_split_ratio, transforms=trans, mode=Mode.TRAIN)

# without augmentation
def get_validation_dataset(cfg_path, valid_split_ratio):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    return ChallengeDataset(cfg_path=cfg_path, valid_split_ratio=valid_split_ratio, transforms=trans, mode=Mode.VALID)




if __name__ == '__main__':
    CONFIG_PATH = '/home/soroosh/Documents/Repositories/deep_learning_challenge/configs/config.json'
    dataset = ChallengeDataset(cfg_path=CONFIG_PATH, valid_split_ratio=0.2, mode=Mode.TRAIN, seed=1)
    # pdb.set_trace()
    # a=5