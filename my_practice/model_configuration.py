"""
configure what's needed before training loop
"""
import torch
import numpy as np

# define device
device = 'cude' if torch.cuda.is_available() else 'cpu'

# set training hyperparameters
lr = 0.1

# fix seeds
np.random.seed(42)
torch.manual_seed(42)

# define model
model = torch.nn.Sequential(torch.nn.Linear(1,1)).to(device)

# define optimiser to look for minima
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# define loss function to minimise
loss_fn = torch.nn.MSELoss(reduction='mean')

# Define a fn that does 1 step in the training
from utils import get_one_training_step_fn
one_training_step_fn = get_one_training_step_fn(model, loss_fn, optimizer)

print(f"Done {__file__.__repr__()}")
