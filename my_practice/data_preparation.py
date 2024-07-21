"""
prepare data that is in the memmory
"""
import torch
# from generate_data import x, y

torch.manual_seed(13)  # Yes, this makes a difference

# define available device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# convert x/y data to pytorch tensors BUT dont send to device
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# Define dataset and dataloader for training
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

# Do the split now on top of the dataset
train_split_ratio = 0.8
n_tot = len(dataset)
n_train = int(n_tot * train_split_ratio)
n_val = n_tot - n_train
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

# Init dataloader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False,  # what to do with incomplete batch
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=16,
    # shuffle=True, # do not use shuffle for reproducibility
    # drop_last=False,  # what to do with incomplete batch
)

# test_variable=10

print(f'Done {__file__.__repr__()}')
