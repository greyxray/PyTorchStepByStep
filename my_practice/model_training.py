"""
training loop
"""

epoch = 1000
losses = []

for e in range(epoch):
    loss = one_training_step_fn(x_train_tensor, y_train_tensor)
    losses.append(loss)


print(f"Done {__file__.__repr__()}")
