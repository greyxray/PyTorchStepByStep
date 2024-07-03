"""
training loop
"""
# import torch
# from data_preparation import device, train_loader, val_loader,test_variable
# from model_configuration import model, one_training_step_fn, one_val_step_fn
# from utils import mini_batches_over_epoch


n_epochs = 200
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    train_epoch_loss = mini_batches_over_epoch(device, train_loader, one_training_step_fn)
    # print(f"epoch_loss: {train_epoch_loss}")
    train_losses.append(train_epoch_loss)

    # VALIDATION - no gradients in validation!
    with torch.no_grad():
        val_epoch_loss = mini_batches_over_epoch(device, val_loader, one_val_step_fn)
        # print(f"epoch_loss: {val_epoch_loss}")
        val_losses.append(val_epoch_loss)

    writer.add_scalars(
        main_tag="loss",
        tag_scalar_dict={
            "training": train_losses[-1],
            "validation": val_losses[-1]},
        global_step=epoch,
    )
writer.close()
# print(f"test_variable: {test_variable}")
print(f"Done {__file__.__repr__()}")

print(model.state_dict())



# # Defines number of epochs
# n_epochs = 1000

# for epoch in range(n_epochs):
#     # Sets model to TRAIN mode
#     model.train()

#     # Step 1 - Computes our model's predicted output - forward pass
#     # No more manual prediction!
#     yhat = model(x_train_tensor)

#     # Step 2 - Computes the loss
#     loss = loss_fn(yhat, y_train_tensor)

#     # Step 3 - Computes gradients for both "a" and "b" parameters
#     loss.backward()

#     # Step 4 - Updates parameters using gradients and the learning rate
#     optimizer.step()
#     optimizer.zero_grad()

# print(model.state_dict())
