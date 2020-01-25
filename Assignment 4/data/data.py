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
import pdb

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
epsilon = 1e-15

class ChallengeDataset(Dataset):
    def __init__(self, csv_path, split_ratio, composer=transforms.Compose(transforms.ToTensor()), mode='train'):
        '''
        Args:
            :split_ratio: a parameter which controls the split between your train and validation data.
        '''
        self.mode = mode
        self.csv_path = csv_path
        self.composer = composer
        self.split_ratio = split_ratio
        self.input_list = pd.read_csv(self.csv_path, delimiter=';')


    def __len__(self):
        '''Returns length of the dataset'''
        return len(self.input_list)

    def __getitem__(self, idx):
        '''
        transforms also here
        '''
        image = self.input_list['filename'][idx]
        image = gray2rgb(image)
        label_temp = np.asarray(self.input_list.loc[self.input_list['filename'] == image])[0, 1:]
        label = np.zeros((3), dtype=int)
        label[0] = label_temp[0]
        label[1] = label_temp[1]
        label[2] = label_temp[2]
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label


    def pos_weight(self):
        '''Calculates a weight for positive examples for each class and returns it as a tensor'''
        w_poly_wafer = torch.tensor(1 / (self.input_list['poly_wafer'].sum() + epsilon))
        w_crack = torch.tensor(1 / (self.input_list['crack'].sum() + epsilon))
        w_inactive = torch.tensor(1 / (self.input_list['inactive'].sum() + epsilon))
        return w_poly_wafer, w_crack, w_inactive



def get_train_dataset():
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
    return ChallengeDataset(csv_path='./train.csv', split_ratio=0.8, composer=trans)

# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(train_mean, train_std)])
    return ChallengeDataset(csv_path='./train.csv', split_ratio=0.2, composer=trans)




if __name__ == '__main__':
    dataset = ChallengeDataset('./train.csv', 0.2)
    pdb.set_trace()
    a=5