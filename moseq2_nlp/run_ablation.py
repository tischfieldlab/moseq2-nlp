import numpy as np
import subprocess

num_seeds = 10

for preprocessing in ['targeted_ablation', 'random_ablation']:
    print(preprocessing)
    for seed in range(num_seeds):
        print(seed)
        name = 'ablation_{}_{}'.format(preprocessing,seed)
        process_str = 'python train.py --name {} --preprocessing {}'.format(name, preprocessing)
        subprocess.call(process_str, shell=True)
