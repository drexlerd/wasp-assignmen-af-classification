import argparse
from pathlib import Path
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import get_dataloaders

from train import eval_loop


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--models_folder", type=str, required=True)
  args = arg_parser.parse_args()

  # set seed
  seed = 0
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tqdm.write("Use device: {device:}\n".format(device=device))

  loss_function = torch.nn.BCEWithLogitsLoss()

  batch_size = 32
  train_dataloader, valid_dataloader, n_classes, len_dataset = get_dataloaders(seed, batch_size)

  models = []
  for model_file in Path(args.models_folder).glob("*.pt"):
    model = torch.load(model_file)
    model.to(device)
    models.append(model)

  all_valid_loss = []
  all_valid_pred = []
  all_valid_true = []
  for model in models:
    valid_loss, valid_pred, valid_true = eval_loop(1, valid_dataloader, model, loss_function, device)
    all_valid_loss.append(valid_loss)
    all_valid_pred.append(valid_pred)
    all_valid_true.append(valid_true)

  assert all(x == all_valid_true[0] for x in all_valid_true)
  
  # Majority vote for the predictions
  preds = []
  for i in range(len(all_valid_pred[0])):
    votes = []
    for j in range(len(all_valid_pred)):
      votes.append(all_valid_pred[j][i])
    preds.append(max(set(votes), key = votes.count))

  valid_auroc = roc_auc_score(all_valid_true[0], preds)
  tqdm.write("Validation AUROC: {:.2f}".format(valid_auroc))

  valid_ap = average_precision_score(all_valid_true[0], preds)
  tqdm.write("Validation AP: {:.2f}".format(valid_ap))


  
