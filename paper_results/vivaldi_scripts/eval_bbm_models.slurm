#!/bin/bash
#SBATCH -o eval_logs/%j.out
#SBATCH --job-name integrators
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB
#SBATCH --time 4-00:00:00

cd /home/solvee/PHNN-PDE/
srun python -u model_evaluation.py --modelfolder 'trained_models' --system 'bbm' --nrollouts 10\
 --dt_rollout .01 --t_max 1