from data.data_handler import get_validation_dataset
import numpy as np
import torch


CONFIG_PATH = '/home/soroosh/Documents/Repositories/deep_learning_challenge/configs/config.json'

train_dl = torch.utils.data.DataLoader(get_validation_dataset(CONFIG_PATH, 0.2), batch_size=1)

a = 0.0
s = np.zeros(3)
s2 = np.zeros(3)
for x, _ in train_dl:
    x = x[0].cpu().numpy()
    a += np.prod(x.shape[1:])
    s += np.sum(x, axis=(1,2))
    s2 += np.sum(x**2, axis=(1,2))

assert x.shape[0] == 3 and x.shape[1] == 300 and x.shape[1] == 300, "your samples are not correctly shaped"

for i in range(3):
    assert s[i] > -a*0.09 and s[i] < a*0.09, "your normalization seems wrong"
    assert s2[i] > a*0.91 and s2[i] < a*1.09, "your normalization seems wrong"

