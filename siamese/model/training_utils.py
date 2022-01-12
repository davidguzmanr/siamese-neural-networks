import random
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, optimizer, device, epoch):
    """
    Trains a single epoch.

    Parameters
    ----------
    model: PyTorch model
        Model to train.

    train_loader: DataLoader.
        PyTorch dataloader with the training data.

    optimizer: torch.optim
        Optmizer to train the model.

    device: torch.device.
        Device where the model will be trained, 'cpu' or 'gpu'.

    epoch: int.
        Current epoch.
    """
    model.train()
    for x1, x2, y in tqdm(train_loader, leave=False, desc=f'Epoch {epoch}'):
        # Move to GPU, in case there is one
        x1, x2, y = x1.to(device), x2.to(device), y.unsqueeze(dim=1).float().to(device)
        
        # Compute logits
        y_lgts = model(x1, x2)
        
        # Compute the loss
        loss = F.binary_cross_entropy_with_logits(y_lgts, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_epoch(model, data_loader, device):
    """
    Evaluates a single epoch.
    
    Parameters
    ----------

    model: PyTorch model.
        Model to train.
    
    data_loader: DataLoader.
        PyTorch dataloader with data to validate.
    
    device: torch.device.
        Device where the model will be trained, 'cpu' or 'gpu'.
    """
    model.eval()
    with torch.no_grad():
        losses, accs = [], []

        for x1, x2, y in tqdm(data_loader, leave=False, desc='Eval'):
            # Move to GPU, in case there is one
            x1, x2, y = x1.to(device), x2.to(device), y.unsqueeze(dim=1).float().to(device)

            # Compute logits
            y_lgts = model(x1, x2)

            # Compute the loss
            loss = F.binary_cross_entropy_with_logits(y_lgts, y)

            # Get the classes
            y_pred = y_lgts.sigmoid().round()

            # Compute accuracy
            accuracy = (y_pred == y).type(torch.float32).mean()

            # Save the current loss and accuracy
            losses.append(loss.item())
            accs.append(accuracy.item())

        # Compute the mean
        loss = np.mean(losses) * 100
        accuracy = np.mean(accs) * 100

        return loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves a checkpoint of the model for the current epoch.
    
    Parameters
    ----------

    model: PyTorch model.
        Model to train.

    optimizer: PyTorch optimizer.
        Optimizer for the model. 

    epoch: int.
        Number of the current epoch.

    loss: float.

        Loss (in train) of the current epoch.

    path: str.
        Path to save the checkpoint.
    """
    if path:
        torch.save(
            {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(), 
             'loss': loss,
             }, path)

def train(model, train_loader, validation_loader, device, lr=1e-3, weight_decay=0.0, 
          epochs=20, patience=3, writer=None, checkpoint_path=None):
    """
    Trains the whole model.

    Parameters
    ----------

    model: PyTorch model.
        Model to train.

    train_loader: DataLoader.
        PyTorch dataloader with the training data.
    
    validation_loader: DataLoader.
        PyTorch dataloader with the validation data.
    
    device: torch.device
        Device where the model will be trained, 'cpu' or 'gpu'
    
    lr: float, default=1e-3.
        Learning rate.

    weight_decay: float, default=0.0.
        Weight decay for L2 penalty.
    
    epochs: int, default=20.
        Number of epochs.
    
    patience: int, default=3.
        Number of epochs with no improvement after which training 
        will be stopped for early stopping.
    
    writer: instance of SummaryWriter, default=None.
        Writer for Tensorboard.
    
    checkpoint_path: str, default=None.
        Path to save checkpoints for the model, if None no checkpoints 
        will be saved.
    
    Notes
    -----
    - See https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
    """
    last_loss, early_stop = np.inf, 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in trange(epochs, desc='Train'):
        # Train a single epoch
        train_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate in training
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        # Evaluate in validation
        val_loss, val_acc = eval_epoch(model, validation_loader, device)

        if writer:
            writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag='Accuracy/train', scalar_value=train_acc, global_step=epoch)

            writer.add_scalar(tag='Loss/validation', scalar_value=val_loss, global_step=epoch)
            writer.add_scalar(tag='Accuracy/validation', scalar_value=val_acc, global_step=epoch)

        # Early stopping
        current_loss = val_loss
        if current_loss > last_loss:
            early_stop += 1
            if early_stop > patience:
                print('Early stopping!!!')
                return # Stop training
        else:
            early_stop = 0
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)

        last_loss = current_loss