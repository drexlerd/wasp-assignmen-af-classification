#!/bin/bash
#
#SBATCH -J ecg_smac
#SBATCH -t 3-00:00:00
#SBATCH -C thin --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dominik.drexler@liu.se,martin.funkquist@liu.se

RUN_ERR="run.err"
RUN_LOG="run.log"

python3 hyperparam_tuning.py --n_workers 12 --n_trials 100 --walltime_limit 172800 2> ${RUN_ERR} 1> ${RUN_LOG}
