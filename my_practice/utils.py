import torch
from typing import Callable

def get_one_training_step_fn(model, loss_fn: Callable, optimizer) -> Callable:
    """
    TODO: redo with partial
    """

    def one_training_step_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        # set model to train mode
        model.train()

        # pred
        y_pred_tensor = model(x)

        # loss -> number?
        loss = loss_fn(y_pred_tensor, y)

        # loss backprop
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return one_training_step_fn


def get_one_val_step_fn(model, loss_fn: Callable) -> Callable:
    """
    TODO: redo with partial
    """

    def one_val_step_fn(x: torch.Tensor, y: torch.Tensor) -> float:
        # set model to eval mode
        model.eval()

        y_pred_tensor = model(x)

        # loss -> number?
        loss = loss_fn(y_pred_tensor, y)

        return loss.item()

    return one_val_step_fn


def mini_batches_over_epoch(device, data_loader, one_step_fn) -> float:
    batch_losses = []
    for x_batch, y_batch in data_loader:
        # important!
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        loss = one_step_fn(x_batch, y_batch)
        batch_losses.append(loss)
    return np.mean(batch_losses)
