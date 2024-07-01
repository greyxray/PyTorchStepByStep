"""
prepare data that is in the memmory
"""
import torch

# define available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# convert x/y train data to pytorch tensors AND send to device
x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)


print(f'Done {__file__.__repr__()}')