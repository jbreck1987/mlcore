"""
This module contains utilities that simplify the training/testing loop for different model
architectures
"""
# Author: Josh Breckenridge
# Date: 7/26/2023

import torch
from typing import Any, Tuple, Dict
from math import isclose

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


def make_predictions(model: torch.nn.Module, samples: list, device: torch.device) -> list:
    """
    Given a list of samples, returns a list with the prediction for each sample
    """
    # Move model and input data to passed device
    model.to(device=device)
    s = [sample.to(device=device) for sample in samples]

    model.eval()
    with torch.inference_mode():
        # This comprehension creates a list of predictions on the samples that have been moved to the passed in device
        # and then moves that prediction back to original device according to the corresponding original sample
        # passed into the function. It then makes sure the returned prediction is a Pytorch tensor.
        # Not very memory efficient as-is, but makes sure the returned predictions can be used in later
        # code without dealing device mismatches.
        return [torch.tensor(model(sample)).to(dev) for sample, dev in zip(s, [x.device for x in samples])]
    


class EarlyStop:
    """
    This is a simple class that stops a run early if they model seems
    to be saturating and/or diverging.

    Inputs:
        -div_threshold (float): The difference in the two metrics that should be considered being divergent
        -sat_tolerance (int): The allowed number of steps for which the minimum value of the tracked
         metric has not been updated
        -div_tolerance (int): The allowed number of steps for which the two tracked metrics have
         been considered divereged (div_threshold met)
        -sat_metric (bool): Which metric to choose to track to determine whether a training session
         has saturated. This can either be 0 or 1, which corressponds to the first
         and second metrics passed into the object when calling.
        -track_sat/div (bool): Determines whether or not to track saturation and/or divergence.
    """

    def __init__(self,
                 sat_tolerance: int = 0,
                 div_tolerance: int = 0,
                 div_threshold: float = 0.5,
                 sat_metric: bool = 0,
                 track_div: bool = False,
                 track_sat: bool = False
                ):
        
        if not (track_sat or track_div):
            raise RuntimeError('Either saturation or divergence tracking needs to be enabled.')

        self.st = sat_tolerance
        self.dt = div_tolerance
        self.s_metric = sat_metric
        self.s_track = track_sat
        self.d_track = track_div
        self.d_threshold = div_threshold
        self.s_min = None
        self.s_counter = 0
        self.d_counter = 0

    # Define methods to determine whether the metric(s) have saturated/diverged
    def _check_saturated(self, metric: float) -> bool:
        if self.s_min == None or metric < self.s_min:
            self.s_min = metric
            self.s_counter = 0
            return False
        elif self.s_counter < self.st:
            self.s_counter += 1
            return False
        else:
            return True
    
    def _check_diverged(self, metric1: float, metric2: float) -> bool:
        if isclose(metric1, metric2, abs_tol=self.d_threshold):
            self.d_counter = 0
            return False
        elif self.d_counter < self.dt:
            self.d_counter += 1
            return False
        else:
            return True
        
    def reset(self) -> None:
        self.s_counter = 0
        self.d_counter = 0
    
    def __call__(self, metric1: float, metric2: float) -> bool:
        sat_res = False
        div_res = False

        if self.s_track:
            if self.s_metric:
                sat_res = self._check_saturated(metric2)
            else:
                sat_res = self._check_saturated(metric1)
        if self.d_track:
            div_res = self._check_diverged(metric1, metric2)
        return sat_res or div_res



