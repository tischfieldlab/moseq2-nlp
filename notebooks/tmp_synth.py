import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from moseq2_nlp.models import DocumentEmbedding
from moseq2_nlp.data import load_groups, get_transition_representation, get_transition_representations_n, sample_markov_chain
from moseq2_nlp.train import train_regressor, train_svm
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import (get_transition_matrix,
                                   parse_model_results,
                                   results_to_dataframe,
                                   relabel_by_usage, get_syllable_statistics)
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pdb
import os
from tqdm import tqdm
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

min_n, max_n = 1,6
min_k, max_k = 1,3
acc_mx = np.zeros((max_n, max_k))
num_syllables_range = [10000,15000]

data_dir = '/cs/labs/mornitzan/ricci/data/abraira'
data_name = '2020-11-10_Celsr3_R774H'
model_file = os.path.join(data_dir, data_name, 'robust_septrans_model_1000.p')
index_file = os.path.join(data_dir, data_name, 'gender-genotype-index.yaml')

# Get nth order transitions (usages, transitions, 3grams, etc.)
n=4
num_transitions=100
max_syllable=25
dm=2
embedding_dim=300
embedding_epochs=50
min_count = 1
scoring = 'accuracy'
K = 1
penalty = 'l2'
num_c = 11
seed = 0

pbar = tqdm((max_n - min_n)*(max_k - min_k), position=0, leave=True)

for n in range(min_n,max_n + 1):
    for k in range(min_k,max_k + 1):
        print(f'Running order {n}, window {k}.')
        embedding_window = k
        
        # synthesize data
        transition_matrices, out_groups = get_transition_representations_n(model_file, index_file, n,
                                                                   num_transitions, normalize='row', max_syllable=max_syllable)
        
        all_synthesized_data = []
        for l, tmx in tqdm(enumerate(transition_matrices)):
            num_syllables = np.random.randint(num_syllables_range[0], num_syllables_range[1])
            all_synthesized_data.append(sample_markov_chain(tmx,num_syllables))
        
        unique_groups = []
        for group in out_groups:
            if group not in unique_groups:
                unique_groups.append(group)

        labels = []
        for group in out_groups:
            labels.append(unique_groups.index(group))
        
        # get embeddings
        de = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window,
                               embedding_epochs=embedding_epochs,min_count=min_count)
        print('Learning embeddings...')
        E = np.array(de.fit_predict(all_synthesized_data))
        
        print('Classifying...')
        # classify
        best_C, scores = train_regressor(E, labels, K, scoring, penalty, num_c, seed)
        
        print('Updating...')
        # store
        acc_mx[n-1,k-1] = np.max(scores.mean((0,1)))
        pbar.update(1)
pbar.close()

np.save('./acc_mx.npy')