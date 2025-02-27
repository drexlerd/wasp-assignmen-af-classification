import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import get_dataloaders
from model import Model

from train import eval_loop


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--models_folder", type=str, required=True)
  args = arg_parser.parse_args()

  # set seed
  seed = 42
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tqdm.write("Use device: {device:}\n".format(device=device))

  loss_function = torch.nn.BCEWithLogitsLoss()

  batch_size = 32

  # hparams = json.load(open(os.path.join(args.models_folder, "hparams.json")))
  model_files = sorted(Path(args.models_folder).glob("*.pth"))
  models = []
  # for model_file, hp in zip(model_files, hparams):
  for model_file in model_files:
    # model = Model(kernel_size=hp["kernel_size"], n_res_blks=hp["n_res_blks"], 
    #               dropout_rate=hp["dropout_rate"], out_channels=hp["out_channels"], factor=1)
    model = torch.load(model_file, map_location=device)
    # model_dict = torch.load(model_file, map_location=device)
    # model.load_state_dict(model_dict["model"], strict=False)
    # model.to(device)
    models.append(model)

  all_valid_loss = []
  all_valid_pred = []
  all_valid_true = []
  for model in models:
    _, valid_dataloader, _, _ = get_dataloaders(seed, batch_size)
    valid_loss, valid_pred, valid_true = eval_loop(1, valid_dataloader, model, loss_function, device)
    all_valid_loss.append(valid_loss)
    all_valid_pred.append(np.squeeze(valid_pred).tolist())
    all_valid_true.append(np.squeeze(valid_true).tolist())

  assert all(all(x == y for x, y in zip(l, all_valid_true[0])) for l in all_valid_true)
  
  # Majority vote for the predictions
  preds = []
  for i in range(len(all_valid_pred[0])):
    votes = []
    for j in range(len(all_valid_pred)):
      votes.append(all_valid_pred[j][i])
    preds.append(max(set(votes), key = votes.count))

  valid_auroc = roc_auc_score(all_valid_true[0], preds)
  tqdm.write("Ensemble Validation AUROC: {:.2f}".format(valid_auroc))

  valid_ap = average_precision_score(all_valid_true[0], preds)
  tqdm.write("Ensemble Validation AP: {:.2f}".format(valid_ap))

  for i, (pred, true) in enumerate(zip(all_valid_pred, all_valid_true)):
    auroc = roc_auc_score(true, pred)
    ap = average_precision_score(true, pred)

    tqdm.write(f"Model {str(i+1)} AUROC: {auroc}")
    tqdm.write(f"Model {str(i+1)} AP: {ap}")




  
