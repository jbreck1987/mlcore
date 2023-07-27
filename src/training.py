"""
This module contains utilities that simplify the training/testing loop for different model
architectures
"""
# Author: Josh Breckenridge
# Date: 7/26/2023

import torch

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device) -> None:

    # Training Steps
    total_loss, acc = 0, 0 # need to reset loss every epoch
    model.to(device=device) # send model to GPU if available
    for batch, (X, y) in enumerate(data_loader): # each batch has 32 data/labels, create object -> (batch, (X, y))
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

        # We want to see some updates within an epoch
        print(f'Batches processed: {batch + 1}/{len(data_loader)}, Samples processed: {(batch + 1) * data_loader.batch_size}/{len(data_loader.dataset)}', end='\r')
    
    # Now we want to find the AVERAGE loss and accuracy of all the batches
    total_loss /= len(data_loader)
    acc /= len(data_loader)
    print('\n-----------')
    print(f'Mean Train Loss: {total_loss:.4f}, Mean Train Accuracy: {(acc * 100):.4f} ')


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device) -> None:

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
    print(f'Mean Test Loss: {total_loss:.4f}, Mean Test Accuracy: {(acc * 100):.4f} ')
    print('-----------\n')


def make_predictions(model: torch.nn.Module, samples: list) -> list:
    """
    Given a list of samples, returns a tensor with the prediction for each sample
    """
    model.eval()
    with torch.inference_mode():
        return [model(x) for x in samples]


