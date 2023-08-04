import numpy as np
import torch

from tqdm.notebook import trange, tqdm
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

from dataset import get_dataloaders
from model import Model
from train import eval_loop, train_loop
import globals

# set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters
learning_rate = 1e-3
weight_decay = 1e-1
batch_size = 4
lr_scheduler = None

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tqdm.write("Use device: {device:}\n".format(device=device))

train_dataloader, valid_dataloader, n_classes, len_dataset = get_dataloaders(seed, batch_size)

# =============== Define model ============================================#
tqdm.write("Define model...")
"""
TASK: Replace the baseline model with your model; Insert your code here
"""
# model = ModelBaseline()
model = Model()
model.to(device=device)
tqdm.write("Done!\n")

# =============== Define loss function ====================================#
loss_function = torch.nn.BCEWithLogitsLoss()

# =============== Define optimizer ========================================#
tqdm.write("Define optimiser...")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
tqdm.write("Done!\n")

# =============== Train model =============================================#
tqdm.write("Training...")
best_loss = np.Inf
# allocation
train_loss_all, valid_loss_all, train_auroc_all, valid_auroc_all = [], [], [], []

# loop over epochs
for epoch in trange(1, globals.n_epochs + 1):
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

    print("train_pred:", train_pred_flat)
    print("valid_pred:", valid_pred_flat)
    print("train_true:", train_true_flat)
    print("valid_true:", valid_true_flat)

    # Calculate AUROC
    train_auroc = roc_auc_score(train_true_flat, train_pred_flat)
    train_auroc_all.append(train_auroc)
    tqdm.write("Training AUROC: {:.4f}".format(train_auroc))

    valid_auroc = roc_auc_score(valid_true_flat, valid_pred_flat)
    valid_auroc_all.append(valid_auroc)
    tqdm.write("Validation AUROC: {:.4f}".format(valid_auroc))

    # save best model: here we save the model only for the lowest validation loss
    if valid_loss < best_loss:
        # Save model parameters
        torch.save({'model': model.state_dict()}, 'model.pth')
        # Update best validation loss
        best_loss = valid_loss
        # statement
        model_save_state = "Best model -> saved"
    else:
        model_save_state = ""

    # Print message
    tqdm.write('Epoch {epoch:2d}: \t'
                'Train Loss {train_loss:.6f} \t'
                'Valid Loss {valid_loss:.6f} \t'
                '{model_save}'
                .format(epoch=epoch,
                        train_loss=train_loss,
                        valid_loss=valid_loss,
                        model_save=model_save_state)
                    )

    # Update learning rate with lr-scheduler
    if lr_scheduler:
        lr_scheduler.step()

# =============== Plot results ============================================
plt.title("Train and Validation Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
train_x = range(len(train_loss_all))
train_y = train_loss_all
plt.plot(train_x, train_y, label="train_loss", color="blue")
valid_x = range(len(valid_loss_all))
valid_y = valid_loss_all
plt.plot(valid_x, valid_y, label="valid_loss", color="orange")
plt.legend()
plt.show()

plt.title("Area Under Curve (AUROC)")
plt.xlabel("epoch")
plt.ylabel("auroc")
train_auroc_x = range(len(train_auroc_all))
train_auroc_y = train_auroc_all
plt.plot(train_auroc_x, train_auroc_y, label="train_auroc", color="blue")
valid_auroc_x = range(len(valid_auroc_all))
valid_auroc_y = valid_auroc_all
plt.plot(valid_auroc_x, valid_auroc_y, label="valid_auroc", color="orange")
plt.show()


