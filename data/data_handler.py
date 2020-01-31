"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

from torch.utils.data import Dataset
import torch
import csv
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
    def __init__(self, cfg_path, valid_split_ratio=0.2, transform=transforms.Compose(transforms.ToTensor()), mode=Mode.TRAIN, seed=42):
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
        self.input_data_path = params['input_data_path']
        self.cfg_path = cfg_path
        self.mode = mode
        self.transform = transform

        self.input_list = []
        self.sum_crack = 0
        self.sum_inactive = 0

        with open(params['input_csv_path']) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=';')
            for row in reader:
                self.input_list.append(row)
                self.sum_crack += int(row['crack'])
                self.sum_inactive += int(row['inactive'])
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

        label = np.zeros((2), dtype=int)
        label[0] = int(output_list[idx]['crack'])
        label[1] = int(output_list[idx]['inactive'])

        # Reads images using files name available in the list
        image = imread(os.path.join(self.input_data_path, output_list[idx]['filename']))
        image = gray2rgb(image)
        image = self.transform(image)
        label = torch.from_numpy(label)
        return image, label


    def pos_weight(self):
        '''Calculates a weight for positive examples for each class and returns it as a tensor'''
        w_crack = torch.tensor((len(self.input_list) - self.sum_crack) / (self.sum_crack + epsilon))
        w_inactive = torch.tensor((len(self.input_list) - self.sum_inactive) / (self.sum_inactive + epsilon))
        return w_crack, w_inactive



def get_train_dataset(cfg_path, valid_split_ratio):
    # since all the images are 300 * 300, we don't need resizing
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
    return ChallengeDataset(cfg_path=cfg_path, valid_split_ratio=valid_split_ratio, transform=trans, mode=Mode.TRAIN)

# without augmentation
def get_validation_dataset(cfg_path, valid_split_ratio):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    return ChallengeDataset(cfg_path=cfg_path, valid_split_ratio=valid_split_ratio, transform=trans, mode=Mode.VALID)




if __name__ == '__main__':
    CONFIG_PATH = '/home/soroosh/Documents/Repositories/deep_learning_challenge/configs/config.json'
    dataset = get_train_dataset(CONFIG_PATH, valid_split_ratio=0.2)
    # pdb.set_trace()
    # a=5