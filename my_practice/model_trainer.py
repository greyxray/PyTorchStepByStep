from typing import Optional
import datetime
import matplotlib.pyplot as plt
import numpy as np

import torch
from  torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# from model_configuration import get_tensorboard_writer

class ModelTrainer:
    """Does model training with model that is owns
        on the data that is passed to it."""

    # conceptually these are not part of the model, but the input
    # since we can specify model w/o dataloader -> placeholders
    train_loader: DataLoader = None
    val_loader: Optional[DataLoader] = None
    writer: Optional[SummaryWriter] = None

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # write to device the model
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device) # by default resolved automatically

        self.train_losses = []
        self.val_losses = []
        self.total_epochs = 0

        '''
        from utils import get_one_training_step_fn, get_one_val_step_fn

        self._one_training_step_fn = get_one_training_step_fn(model, loss_fn, optimizer)
        self._one_val_step_fn = get_one_val_step_fn(model, loss_fn)
        '''

        # self.checkpoint = {}

    def to(self, device):
        try:
            self.model.to(device)
            self.device = device
        except RuntimeError as exc:
            print(f"Couldn't save model to {device} will keep using {self.device}")

    # fn to set dataloaders
    def set_loaders(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

    # fn to write to tensorboard
    def set_tensorboard(self, experiment_name: str, folder: str = "runs"):
        # print(f"Will log experiment to {experiment_name}")
        timestamp = datetime.datetime.now().now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter(f"{folder}/{experiment_name}_{timestamp}")

    def set_tensorboard_graph(self):
        if self.writer:
            if self.train_loader:
                dummy_x, _ = next(iter(self.train_loader))
                self.writer.add_graph(self.model, dummy_x.to(self.device))
            else:
                print("No train_loader configured")
        else:
            print("No tensorboard writer configured")

    def save_checkpoint(self, filename="model_checkpoint.pth"):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_epochs": self.total_epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename="model_checkpoint.pth"):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["total_epochs"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]

    def _one_training_step_fn(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # set model to train mode
        self.model.train()

        # pred
        y_pred_tensor = self.model(x)

        # loss -> number?
        loss = self.loss_fn(y_pred_tensor, y)

        # loss backprop
        loss.backward()

        # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def _one_val_step_fn(self, x: torch.Tensor, y: torch.Tensor) -> float:
        # set model to eval mode
        self.model.eval()

        y_pred_tensor = self.model(x)

        # loss -> number?
        loss = self.loss_fn(y_pred_tensor, y)

        return loss.item()

    def _mini_batches_over_epoch(self, validation: bool = False) -> float:
        """
        Runs mini-batches over 1 epoch
        """
        if validation:
            data_loader = self.val_loader
            one_step_fn = self._one_val_step_fn
            self.model.eval()
        else:
            data_loader = self.train_loader
            one_step_fn = self._one_training_step_fn
            self.model.train()

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            # important!
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            loss = one_step_fn(x_batch, y_batch)
            mini_batch_losses.append(loss)

        return np.mean(mini_batch_losses)

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.as_tensor(x).float().to(self.device)  # ????: is the memory on device freed when exiting the fn?
        y_hat_tensor = self.model(x_tensor)

        return (
            y_hat_tensor.detach().cpu().numpy()
        )  # ????: is the memory on device freed when exiting the fn?

    @staticmethod
    def set_seed(seed):
        """
        does this work with static?
        """
        torch.backends.cudnn.deterministic = True # ????
        torch.backends.cudnn.benchmark = False  # ????
        torch.manual_seed(seed)
        np.random.seed(seed)

    def run_training(self, n_epochs, seed=42):
        """
        run over epochs
        """
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # train 1 epoch
            train_epoch_loss = self._mini_batches_over_epoch(validation=False)
            self.train_losses.append(train_epoch_loss)

            # eval 1 epoch
            # VALIDATION - no gradients in validation!
            with torch.no_grad():
                val_epoch_loss = self._mini_batches_over_epoch(validation=True)
                self.val_losses.append(val_epoch_loss)

            if self.writer:
                scalars = {"training": train_epoch_loss}
                if val_epoch_loss:
                    scalars["validation"] = val_epoch_loss

                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch,
                )

            self.total_epochs += 1

        if self.writer:
            # self.writer.close()
            self.writer.flush()  # ?????

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))

        plt.plot(self.train_losses, label='Training Loss', c='b')
        if self.val_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')

        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        return fig
