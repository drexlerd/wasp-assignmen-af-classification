import argparse
import os
from pathlib import Path
import numpy as np
import torch

from ConfigSpace import ConfigurationSpace
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from multiprocessing import freeze_support

from train import train


if __name__ == "__main__":
  freeze_support()
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--n_workers", type=int, default=2)
  arg_parser.add_argument("--n_trials", type=int, default=4)
  arg_parser.add_argument("--n_epochs", type=int, default=200)
  arg_parser.add_argument("--walltime_limit", type=int, default=120)
  arg_parser.add_argument("--output_folder", type=str)
  
  args = arg_parser.parse_args()

  cs = ConfigurationSpace({
    "lr": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "kernel_size": [7, 15, 31, 63],
    "n_res_blks": [1, 2, 3, 4],
    "dropout_rate": [0.0, 0.5, 0.8],
    "batch_size": [16, 32, 64],
    "out_channels": [8, 16, 32, 64, 128],
    "factor": [2, 4, 8],
    "n_epochs": [args.n_epochs]
  })

  # set seed
  seed = 42
  np.random.seed(seed)
  torch.manual_seed(seed)

  scenario = Scenario(
      configspace=cs,
      output_directory=Path(args.output_folder),
      walltime_limit=args.walltime_limit,
      # trial_walltime_limit=10,  # Limit to 30 seconds per trial
      n_trials=args.n_trials,  # Evaluated max 500 trials
      # n_workers=8,  # Use eight workers
      n_workers=args.n_workers,
  )


  smac = HPOFacade(scenario=scenario, target_function=train)
  incumbent = smac.optimize()

  with open(os.path.join(args.output_folder, "incumbent.txt"), "w") as f:
    f.write(str(incumbent))
