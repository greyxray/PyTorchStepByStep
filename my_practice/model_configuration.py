"""
configure what's needed before training loop
"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils import get_one_training_step_fn, get_one_val_step_fn

# define device
device = 'cude' if torch.cuda.is_available() else 'cpu'

# set training hyperparameters
lr = 0.1

# fix seeds
# np.random.seed(42)
torch.manual_seed(42)

# define model
model = torch.nn.Sequential(torch.nn.Linear(1,1)).to(device)

# define optimiser to look for minima
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# define loss function to minimise
loss_fn = torch.nn.MSELoss(reduction='mean')

# Define a fn that does 1 step in the training
one_training_step_fn = get_one_training_step_fn(model, loss_fn, optimizer)
one_val_step_fn = get_one_val_step_fn(model, loss_fn)


def get_tensorboard_writer(model, data_loader, experiment_name):
    # Tensorboard writer setup
    print(f"Will log experiment to {experiment_name}")
    writer = SummaryWriter(f"runs/{experiment_name}")
    # Add DAG
    dummy_x, _ = next(iter(data_loader))
    writer.add_graph(model, dummy_x.to(device))

    return writer
writer = get_tensorboard_writer(model, train_loader, experiment_name="simple_linear_regression")


print(f"Done {__file__.__repr__()}")
