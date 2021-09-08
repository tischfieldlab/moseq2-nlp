import numpy as np
import subprocess

emissions = [True, False]
embedding_window = [2, 4, 8, 16, 32, 64]
embedding_dim = np.logspace(np.log10(70), np.log10(4900),5)
embedding_epochs = [50, 100, 150, 200, 250]
embedding_dim = [int(ed) for ed in embedding_dim]

fn = 'grid_search_commands.txt'
with open(fn,'a') as file:
    for e in emissions:
        for ew in embedding_window:
            for ed in embedding_dim:
                for ee in embedding_epochs:
                    name = 'carageenan_{}_{}_{}_{}'.format(e,ew,ed,ee)
                    process_str = 'python train.py --name {} --representation embeddings --emissions {} --embedding_window {} --embedding_dim {} --embedding_epochs {}'.format(name, e, ew, ed, ee)
                    file.write(process_str + '\n')

                    #subprocess.call('python train.py --name nt_{} --representation embeddingss --num_transitions {}&'.format(nt, nt),shell=True) 
