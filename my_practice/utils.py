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