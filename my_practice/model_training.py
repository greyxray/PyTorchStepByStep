"""
training loop
"""
from data_preparation import *
from model_configuration import *

epoch = 1000
losses = []

for e in range(epoch):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        # important!
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = one_training_step_fn(x_batch, y_batch)
        batch_losses.append(loss)

    losses.append(np.mean(batch_losses))

print(f"Done {__file__.__repr__()}")

print(model.state_dict())
