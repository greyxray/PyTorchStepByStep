"""
prepare data that is in the memmory
"""
import torch
from generate_data import *

# define available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# convert x/y train data to pytorch tensors BUT dont send to device
x_train_tensor = torch.as_tensor(x_train).float()
y_train_tensor = torch.as_tensor(y_train).float()

# Define dataset and dataloader for training
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False,  # what to do with incomplete batch
)

print(f'Done {__file__.__repr__()}')
