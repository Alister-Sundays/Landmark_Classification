import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingLR


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """

    if torch.cuda.is_available():
        model = model.to('cuda')
        model.train()
    
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()# YOUR CODE HERE:
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output  = model(data) # YOUR CODE HERE
        # 3. calculate the loss
        loss_value  = loss(output, target) # YOUR CODE HERE
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()# YOUR CODE HERE:
        # 5. perform a single optimization step (parameter update)
        optimizer.step()# YOUR CODE HERE:

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():
        model.eval()

        # set the model to evaluation mode
        # YOUR CODE HERE

        if torch.cuda.is_available():
            model.cuda()

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data)# YOUR CODE HERE
            # 2. calculate the loss
            loss_value  = loss(output,target)# YOUR CODE HERE

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False, early_stopping_patience=20, scheduler_type='ReduceLROnPlateau'):
  """
  Performs training with early stopping and different learning rate schedulers.

  Args:
      data_loaders (dict): Dictionary containing training and validation data loaders.
      model (nn.Module): The model to be trained.
      optimizer (optim.Optimizer): The optimizer used for training.
      loss (nn.Module): The loss function used for training.
      n_epochs (int): The number of epochs to train for.
      save_path (str): The path to save the model with the best validation loss.
      interactive_tracking (bool, optional): Whether to enable interactive tracking of losses and learning rate. Defaults to False.
      early_stopping_patience (int, optional): The number of epochs to wait for improvement in validation loss before stopping. Defaults to 10.
      scheduler_type (str, optional): The type of learning rate scheduler to use. Can be 'ReduceLROnPlateau', 'OneCycleLR', or 'CosineAnnealingLR'. Defaults to 'ReduceLROnPlateau'.

  Returns:
      dict: A dictionary containing training logs.
  """

  # initialize tracker for minimum validation loss
  if interactive_tracking:
      liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
  else:
      liveloss = None

  valid_loss_min = None
  logs = {}

  # Learning rate scheduler setup
  if scheduler_type == 'ReduceLROnPlateau':
      scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
  elif scheduler_type == 'OneCycleLR':
      # Requires additional parameters like max_lr, steps_per_epoch, and epochs
      raise NotImplementedError("OneCycleLR requires additional parameters. Please specify them during its creation.")
  elif scheduler_type == 'CosineAnnealingLR':
      # Requires additional parameters like T_max (number of epochs for one cycle) and eta_min (minimum learning rate)
      raise NotImplementedError("CosineAnnealingLR requires additional parameters. Please specify them during its creation.")
  else:
      raise ValueError(f"Invalid scheduler_type: {scheduler_type}. Supported options are 'ReduceLROnPlateau', 'OneCycleLR', or 'CosineAnnealingLR'.")

  for epoch in range(1, n_epochs + 1):

      train_loss = train_one_epoch(
          data_loaders["train"], model, optimizer, loss
      )

      valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

      # print training/validation statistics
      print(
          "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
              epoch, train_loss, valid_loss
          )
      )

      # Early stopping and model saving
      if valid_loss_min is None or (
              (valid_loss_min - valid_loss) / valid_loss_min > 0.01
      ):
          print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

          torch.save(model.state_dict(), save_path)  # YOUR CODE HERE

          valid_loss_min = valid_loss

      else:
          # Early stopping check
          early_stopping_counter = early_stopping_counter + 1 if 'early_stopping_counter' in locals() else 1
          if early_stopping_counter >= early_stopping_patience:
              print(f"Early stopping triggered after {early_stopping_counter} epochs with no improvement.")
              break

      # Update learning rate based on scheduler type
      scheduler.step(valid_loss)  # Works for ReduceLROnPlateau

      # Log the losses and the current learning rate
      if interactive_tracking:
          logs["loss"] = train_loss
          logs["val_loss"] = valid_loss
          logs["lr"] = optimizer.param_groups[0]["lr"]

          liveloss.update(logs)
          liveloss.send()

  return logs


def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():
        model.eval()

        # set the model to evaluation mode
        # YOUR CODE HERE

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = model(data)# YOUR CODE HERE
            # 2. calculate the loss
            loss_value  = loss(logits, target)# YOUR CODE HERE

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred  = torch.argmax(logits, dim=1)# YOUR CODE HERE

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
