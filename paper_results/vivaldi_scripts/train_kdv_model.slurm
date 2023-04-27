#!/bin/bash
#SBATCH -o logs/%j.out
#SBATCH --job-name kdv
#SBATCH --partition sintef
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --time 4-00:00:00

cd /home/solvee/PHNN-PDE/
srun python -u train_model.py --system $1 --baseline $2 --ntrainingpoints $3 --sampling_time $4 --t_max $5\
 --F_timedependent $6 --F_spacedependent $7 --F_statedependent $8 --kernel_size $9 --hidden_dim ${10} --integrator 'midpoint' --epochs 50000\
 --l1_param_forces 0.5 --l1_param_dissipation 0. --ntrajectories_val 0 --storedir 'trained_models_kdv' -s -v