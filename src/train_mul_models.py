import argparse, os
import json
from ConfigSpace import ConfigurationSpace
import numpy as np
import torch

from train import train
from get_best_setups import get_best_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_setups_file", type=str)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--hparam_tuning_output_folder", type=str, help="Folder containing the output of the hyperparameter tuning process.")
    args = parser.parse_args()

    # set seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)


    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if not args.best_setups_file and not args.hparam_tuning_output_folder:
        raise ValueError("Either specify '--best_setups_file' or '--hparam_tuning_output_folder'.")
    if args.best_setups_file and args.hparam_tuning_output_folder:
        raise ValueError("Specify either '--best_setups_file' or '--hparam_tuning_output_folder', not both.")

    if args.best_setups_file:
        # load best setups
        best_setups = json.load(open(args.best_setups_file, "r"))
    else:
        # load best setups from hyperparameter tuning
        best_setups = get_best_configs(args.hparam_tuning_output_folder, args.n_best)

    configs = []
    for i, setup in enumerate(best_setups):
        configs.append({
            **setup,
            "id": i+1,
            "save": 1,
            "output_dir": args.output_folder,
        })
    
    for config in configs:
        train(config, 0)
    
    print("Finished training all best setups.")

