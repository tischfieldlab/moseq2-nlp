import numpy as np
import subprocess

num_transitions = np.logspace(np.log10(70), np.log10(4900),2)
num_transitions = [int(nt) for nt in num_transitions]
for nt in num_transitions:
   subprocess.call('python train.py --name nt_{} --representation transitions --num_transitions {}&'.format(nt, nt),shell=True) 
