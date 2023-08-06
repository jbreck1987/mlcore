"""
This module contains utilities that simplify the training/testing loop for different model
architectures
"""
# Author: Josh Breckenridge
# Date: 7/26/2023

import torch
from typing import Tuple, Dict

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device) -> dict:

    # Training Steps
    total_loss, acc = 0, 0 # need to reset loss every epoch
    model.to(device=device) # send model to GPU if available
    for X, y in data_loader: # each batch has 32 data/labels, create object -> (batch, (X, y))
        X, y = X.to(device), y.to(device) # send data to GPU if available

        model.train()
        y_pred = model(X) # Like before, need to get model's predictions
        loss = loss_fn(y_pred, y) # calculate loss for this batch
        total_loss += loss # add loss from this batch (mean loss of 32 samples) to total loss for the epoch (sum of all batch loss)
        acc += accuracy_fn(y_pred, y)

        # backprop step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Now we want to find the AVERAGE loss and accuracy of all the batches
    total_loss /= len(data_loader)
    acc /= len(data_loader)
    return {'loss': total_loss.item(), 'acc': acc}


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device) -> dict:

    # Test Steps
    model.to(device=device) # send model to GPU if available
    model.eval()
    total_loss, acc = 0, 0 # need to reset loss every epoch
    with torch.inference_mode():
        for X, y in data_loader: # each batch has 32 data/labels, create object -> (batch, (X_train, y_train))
            X, y = X.to(device), y.to(device)# send data to GPU if available
            
            y_pred = model(X) # Like before, need to get model's predictions
            loss = loss_fn(y_pred, y) # calculate loss for this batch
            total_loss += loss # add loss from this batch (mean loss of 32 samples) to total loss for the epoch (sum of all batch loss)
            acc += accuracy_fn(y_pred, y)


        # Now we want to find the AVERAGE loss and accuracy of all the batches
        total_loss /= len(data_loader)
        acc /= len(data_loader)
    return {'loss': total_loss.item(), 'acc': acc}


def multi_loss_train_step(model: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          loss_fn: Tuple[torch.nn.Module],
                          optimizer: torch.optim.Optimizer,
                          accuracy_fn,
                          device: torch.device) -> dict:
    """
    Performs the train step during model training in the scenario where a model
    needs more than one loss function. Assumes that the model will return a tuple
    of outputs. The order in the model outputs should match the order of the passed in
    loss functions and the indices of the target values in the target tensor.

    E.g. "out_a, out_b = model(X) implies loss_fns=(loss_a, loss_b) and y[0] = target_a,
    y[1] = target_b"

    Returns
        -dict: Contains dicts of the loss value and accuracy for each loss function passed in for
        this epoch
            E.g. {'0':{'loss': loss_val_a, 'acc':acc_a}, '1':{'loss': loss_val_b, 'acc':acc_b}}
    """

    # Training Steps
    # Create tensors to store the accumulating loss and accuracy values for
    # this epoch.
    total_loss = torch.zeros(len(loss_fn), requires_grad=False, device=device)
    acc = torch.zeros(len(loss_fn), requires_grad=False, device=device)
    model.to(device=device) # send model to GPU if available
    for X, y in data_loader:
        X, y = X.to(device), y.to(device) # send data to GPU if available
        model.train()
        y_preds = model(X) # Like before, need to get model's predictions

        optimizer.zero_grad()
        # Loop over all the loss functions and calculate the associated quantities
        for idx, loss in enumerate(loss_fn):
            l = loss(y_preds[idx], y[:, :, idx]) # Calculate loss
            total_loss[idx] += l.item() # Save loss to return
            acc[idx] += accuracy_fn(y_preds[idx], y[idx]) # Calc and save accuracy for return
            l.backward()

        # backprop step
        optimizer.step()
    
    # Now we want to find the AVERAGE loss and accuracy of all the batches
    total_loss /= len(data_loader)
    acc /= len(data_loader)

    # Return the metrics for each loss function
    return {f'{idx}': {'loss': total_loss[idx].item(), 'acc': acc[idx].item()} for idx in range(len(loss_fn))}

def multi_loss_test_step(model: torch.nn.Module,
                        data_loader: torch.utils.data.DataLoader,
                        loss_fn: Tuple[torch.nn.Module],
                        accuracy_fn,
                        device: torch.device) -> dict:
    """
    Performs the test step during model training in the scenario where a model
    needs more than one loss function. Assumes that the model will return a tuple
    of outputs. The order in the model outputs should match the order of the passed in
    loss functions and the indices of the target values in the target tensor.

    E.g. "out_a, out_b = model(X) implies loss_fns=(loss_a, loss_b) and y[0] = target_a,
    y[1] = target_b"

    Returns
        -dict: Contains dicts of the loss value and accuracy for each loss function passed in for
        this epoch
            E.g. {'0':{'loss': loss_val_a, 'acc':acc_a}, '1':{'loss': loss_val_b, 'acc':acc_b}}
    """

    # Test Steps
    model.to(device=device) # send model to GPU if available
    model.eval()
    
    # Create tensors to store the accumulating loss and accuracy values for
    # this epoch.
    total_loss = torch.zeros(len(loss_fn), requires_grad=False, device=device)
    acc = torch.zeros(len(loss_fn), requires_grad=False, device=device)
    with torch.inference_mode():
        for X, y in data_loader: # each batch has 32 data/labels, create object -> (batch, (X_train, y_train))
            X, y = X.to(device), y.to(device)# send data to GPU if available
            
            y_preds = model(X) # Like before, need to get model's predictions

            # Loop over all the loss functions and calculate the associated quantities
            for idx, loss in enumerate(loss_fn):
                l = loss(y_preds[idx], y[:, :, idx]) # Calculate loss
                total_loss[idx] += l.item() # Save loss to return
                acc[idx] += accuracy_fn(y_preds[idx], y[idx]) # Calc and save accuracy for return

        # Now we want to find the AVERAGE loss and accuracy of all the batches
        total_loss /= len(data_loader)
        acc /= len(data_loader)

    # return the metrics for each loss function
    return {f'{idx}': {'loss': total_loss[idx].item(), 'acc': acc[idx].item()} for idx in range(len(loss_fn))}


def make_predictions(model: torch.nn.Module, samples: list) -> list:
    """
    Given a list of samples, returns a tensor with the prediction for each sample
    """
    model.eval()
    with torch.inference_mode():
        return [torch.tensor(model(x)) for x in samples]


