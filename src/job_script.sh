#!/bin/bash
#
#SBATCH -J ecg_smac
#SBATCH -t 3-00:00:00
#SBATCH -C thin --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dominik.drexler@liu.se,martin.funkquist@liu.se

python3 hyperparam_tuning.py --num_workers 32 --num_trials 100 --walltime_limit 172800
