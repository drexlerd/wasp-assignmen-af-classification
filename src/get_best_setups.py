import argparse
import os
from pathlib import Path
import numpy as np
import torch
import json

from ConfigSpace import ConfigurationSpace
from callback import CustomCallback
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac.runhistory import RunHistory

from train import train


def get_cs_dict(configspace_file):
  saved_cs = json.load(open(configspace_file, "r"))
  cs_dict = {}
  for item in saved_cs["hyperparameters"]:
    cs_dict[item["name"]] = item["choices"]
  return cs_dict

def get_best_configs(input_folder: str, n_best: int):
  # load runhistory
  runhistory_file = os.path.join(input_folder, "runhistory.json")
  configspace_file = os.path.join(input_folder, "configspace.json")
  cs_dict = get_cs_dict(configspace_file)
  cs = ConfigurationSpace(cs_dict)
  runhistory = RunHistory()
  runhistory.load(runhistory_file, cs)

  # get best configurations
  all_configs = runhistory.get_configs(sort_by="cost")
  best_configs = all_configs[:n_best]
  # best_config_costs = [runhistory.get_cost(config) for config in best_configs]
  best_configs_json = [dict(config) for config in best_configs]
  return best_configs_json


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--input_folder", type=str)
  arg_parser.add_argument("--output_file", type=str)
  arg_parser.add_argument("--n_best", type=int, default=5)
  args = arg_parser.parse_args()

  best_configs_json = get_best_configs(args.input_folder, args.n_best)  

  with open(args.output_file, "w") as f:
    json.dump(best_configs_json, f, indent=2)
    print("Saved best configurations to", args.output_file)