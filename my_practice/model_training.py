"""
training loop
"""

# import torch
# from data_preparation import device, train_loader, val_loader,test_variable
# from model_configuration import model, one_training_step_fn, one_val_step_fn
# from utils import mini_batches_over_epoch


epoch = 1000
train_losses = []
val_losses = []

for _ in range(epoch):
    train_epoch_loss = mini_batches_over_epoch(device, train_loader, one_training_step_fn)
    # print(f"epoch_loss: {train_epoch_loss}")
    train_losses.append(train_epoch_loss)

    # VALIDATION - no gradients in validation!
    with torch.no_grad():
        val_epoch_loss = mini_batches_over_epoch(device, val_loader, one_val_step_fn)
        # print(f"epoch_loss: {val_epoch_loss}")
        val_losses.append(val_epoch_loss)

# print(f"test_variable: {test_variable}")
print(f"Done {__file__.__repr__()}")

print(model.state_dict())
