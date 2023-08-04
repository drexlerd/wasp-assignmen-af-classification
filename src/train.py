
import random
import numpy as np
from torch import nn
import torch
from ConfigSpace import Configuration
from dataset import get_dataloaders
from sklearn.metrics import roc_auc_score
# from tqdm.notebook import tqdm
from tqdm import tqdm, trange

from model import Model

debug = True

def train_loop(epoch, dataloader, model, optimizer, loss_function, device):
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    train_pred, train_true = [], []
    # progress bar def
    train_pbar = tqdm(dataloader, desc="Training Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    # training loop
    for traces, diagnoses_cpu in train_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces.to(device), diagnoses_cpu.to(device)

        optimizer.zero_grad()
        output = model(traces)
        loss = loss_function(output, diagnoses)
        loss.backward()
        optimizer.step()

        pred_classes = (nn.Sigmoid()(output) > 0.5).float().cpu()
        train_pred.append(pred_classes)
        train_true.append(diagnoses_cpu)

        # Update accumulated values
        total_loss += loss.item()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})

        if debug: break

    train_pbar.close()
    return total_loss / n_entries, np.vstack(train_pred), np.vstack(train_true)

def eval_loop(epoch, dataloader, model, loss_function, device):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    valid_pred, valid_true = [], []
    # progress bar def
    eval_pbar = tqdm(dataloader, desc="Evaluation Epoch {epoch:2d}".format(epoch=epoch), leave=True)
    # evaluation loop
    with torch.no_grad():
        for traces_cpu, diagnoses_cpu in eval_pbar:
            # data to device (CPU or GPU if available)
            traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)

            output = model(traces)
            loss = loss_function(output, diagnoses)
            pred_classes = (nn.Sigmoid()(output) > 0.5).float().cpu()
            valid_pred.append(pred_classes)
            valid_true.append(diagnoses_cpu)

            # Update accumulated values
            total_loss += loss.item()
            n_entries += len(traces)

            # Update progress bar
            eval_pbar.set_postfix({'loss': total_loss / n_entries})

            if debug: break

    eval_pbar.close()
    return total_loss / n_entries, np.vstack(valid_pred), np.vstack(valid_true)


def train(config: Configuration, seed: int) -> float:
  # Hyperparameters
  weight_decay = 1e-1
  batch_size = 32

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tqdm.write("Use device: {device:}\n".format(device=device))

  # =============== Define loss function ====================================#
  loss_function = torch.nn.BCEWithLogitsLoss()

  # =============== Get dataloaders ========================================#
  train_dataloader, valid_dataloader, n_classes, len_dataset = get_dataloaders(seed, batch_size)

  # =============== Define model ============================================#
  model = Model()
  model.to(device=device)

  # =============== Define optimizer ========================================#
  tqdm.write("Define optimiser...")
  optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
  tqdm.write("Done!\n")

  # =============== Train model =============================================#
  tqdm.write("Training...")
  best_loss = np.Inf
  train_loss_all, valid_loss_all, train_auroc_all, valid_auroc_all = [], [], [], []

  # loop over epochs
  for epoch in trange(1, 1 + 1):
      # training loop
      train_loss, y_train_pred, y_train_true = train_loop(epoch, train_dataloader, model, optimizer, loss_function, device)
      # validation loop
      valid_loss, y_valid_pred, y_valid_true = eval_loop(epoch, valid_dataloader, model, loss_function, device)

      # collect losses
      train_loss_all.append(train_loss)
      valid_loss_all.append(valid_loss)

      # Flatten the probabilities and true labels for AUROC calculation
      train_pred_flat = y_train_pred.flatten()
      valid_pred_flat = y_valid_pred.flatten()
      train_true_flat = y_train_true.flatten()
      valid_true_flat = y_valid_true.flatten()

      # Calculate AUROC
      train_auroc = roc_auc_score(train_true_flat, train_pred_flat)
      train_auroc_all.append(train_auroc)
      tqdm.write("Training AUROC: {:.4f}".format(train_auroc))

      valid_auroc = roc_auc_score(valid_true_flat, valid_pred_flat)
      valid_auroc_all.append(valid_auroc)
      tqdm.write("Validation AUROC: {:.4f}".format(valid_auroc))

      # save best model: here we save the model only for the lowest validation loss
      # if valid_loss < best_loss:
      #     # Save model parameters
      #     torch.save({'model': model.state_dict()}, 'model.pth')
      #     # Update best validation loss
      #     best_loss = valid_loss
      #     # statement
      #     model_save_state = "Best model -> saved"
      # else:
      #     model_save_state = ""

      # Print message
      tqdm.write('Epoch {epoch:2d}: \t'
                  'Train Loss {train_loss:.6f} \t'
                  'Valid Loss {valid_loss:.6f} \t'
                  '{model_save}'
                  .format(epoch=epoch,
                          train_loss=train_loss,
                          valid_loss=valid_loss,
                          model_save="test")
                      )

  return 1 - max(valid_auroc_all)  # SMAC always minimizes (the smaller the better)
