import os
import time

system_type = 'kdv'

#n_training_points = [1000, 2000, 5000, 10000, 20000]
n_training_points = [200]
sampling_times = [0.02]
baseline = [2]
F_timedependent = 0
F_spacedependent = 1
F_statedependent = [0]
t_max = .02
kernel_sizes = '[1,3,1,1]'

for b in baseline:
    for n in n_training_points:
        for dt in sampling_times:
            for statedep in F_statedependent:
                if baseline==1 and F_statedependent:
                    dim = 150
                else:
                    dim = 100
                for _ in range(1):
                    s = ('sbatch train_kdv_model.slurm %s %d %d %f %f %d %d %d %s %d'
                            % (system_type, b, n, dt, t_max, F_timedependent, F_spacedependent, statedep, kernel_sizes, dim))
                    print(s)
                    os.system(s)
                    time.sleep(1)
