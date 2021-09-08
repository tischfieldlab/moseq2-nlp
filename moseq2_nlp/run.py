import numpy as np
import subprocess


fn = 'grid_search_commands.txt'

# USAGES
with open(fn,'a') as file:
    name = 'carrageenan_usages'
    process_str = 'python train.py --name {} --representation usages'.format(name)
    file.write(process_str + '\n')

# TRANSITIONS
num_transitions = np.logspace(np.log10(70), np.log10(4900),8)
num_transitions = [int(nt) for nt in num_transitions]
with open(fn,'a') as file:
    for nt in num_transitions:
        name = 'carrageenan_transitions_{}'.format(nt)
        process_str = 'python train.py --name {} --representation transitions --num_transitions {}'.format(name, nt)
        file.write(process_str + '\n')
 
# EMBEDDINGS
emissions = [True, False]
embedding_window = [2, 4, 8, 16, 32, 64]
embedding_dim = np.logspace(np.log10(70), np.log10(4900),8)
embedding_epochs = [50, 100, 150, 200, 250]
embedding_dim = [int(ed) for ed in embedding_dim]

with open(fn,'a') as file:
    for e in emissions:
        for ew in embedding_window:
            for ed in embedding_dim:
                for ee in embedding_epochs:
                    name = 'carrageenan_embeddings_{}_{}_{}_{}'.format(e,ew,ed,ee)
                    process_str = 'python train.py --name {} --representation embeddings --emissions {} --embedding_window {} --embedding_dim {} --embedding_epochs {}'.format(name, e, ew, ed, ee)
                    file.write(process_str + '\n')
