"""
training loop
"""

epoch = 100

for e in range(epoch):
    # set model to train mode
    model.train()

    # pred
    y_pred_tensor = model(x_train_tensor)

    # loss -> number?
    loss = loss_fn(y_pred_tensor, y_train_tensor)

    # loss backprop
    loss.backward()

    # update parameters
    optimizer.step()
    optimizer.zero_grad()


print(f"Done {__file__.__repr__()}")
