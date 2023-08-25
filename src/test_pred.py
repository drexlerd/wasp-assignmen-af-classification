import argparse
import json
import os
import h5py
from torch.utils.data import TensorDataset, random_split, DataLoader

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
  arg_parser.add_argument("--models_folder", type=str)
  arg_parser.add_argument("--model_path", type=str)
  arg_parser.add_argument("--output_file", type=str)
  args = arg_parser.parse_args()

  if not args.models_folder and not args.model_path:
    raise ValueError("Either specify '--models_folder' or '--model_path'.")

  # set seed
  seed = 42
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tqdm.write("Use device: {device:}\n".format(device=device))

  batch_size = 32

  if args.model_path:
    # =============== Build data loaders ==========================================#
    tqdm.write("Building data loaders...")
    # load data
    path_to_h5_test, path_to_csv_test = 'codesubset/test.h5', 'codesubset/test.csv'
    traces = torch.tensor(h5py.File(path_to_h5_test, 'r')['tracings'][()], dtype=torch.float32)
    dataset = TensorDataset(traces)
    len_dataset = len(dataset)
    # build data loaders
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tqdm.write("Done!\n")
    # =============== Define model ================================================#
    tqdm.write("Define model...")

    # load stored model parameters
    model_file = "models/model3.pth"
    model = torch.load(model_file, map_location=device)
    # ckpt = torch.load('model.pth', map_location=lambda storage, loc: storage)
    # model.load_state_dict(ckpt['model'])
    # put model on device
    # model.to(device=device)
    tqdm.write("Done!\n")

    # =============== Evaluate model ==============================================#
    model.eval()
    # allocation
    test_pred = torch.zeros(len_dataset,1)
    # progress bar def
    test_pbar = tqdm(test_dataloader, desc="Testing")
    # evaluation loop
    end=0
    for traces in test_pbar:
        # data to device
        traces = traces[0].to(device)
        start = end
        with torch.no_grad():
            # Forward pass
            model_output = model(traces)

            # store output
            end = min(start + len(model_output), test_pred.shape[0])
            test_pred[start:end] = torch.nn.Sigmoid()(model_output).detach().cpu()

    test_pbar.close()

    # =============== Save predictions ============================================#
    soft_pred = np.stack((1-test_pred.numpy(), test_pred.numpy()),axis=1).squeeze()

    json.dump(soft_pred.tolist(), open(args.output_file, "w"))

    print("Saved predictions to:", args.output_file)

  if args.models_folder:
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




  
