from typing import Dict, List, Literal, Optional, Union
#from pomegranate import DiscreteDistribution, MarkovChain
import numpy as np
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from gensim.models.phrases import original_scorer
from tqdm import tqdm
import pdb
from moseq2_nlp.models import DocumentEmbedding
from scipy.sparse import coo_matrix
import pickle

def load_groups(index_file: str, custom_groupings: List[str]) -> Dict[str, str]:
    # Get group names available in model
    _, sorted_index = parse_index(index_file)
    available_groups = list(set([sorted_index['files'][uuid]['group'] for uuid in sorted_index['files'].keys()]))

    # { subgroup: supergroup }
    group_mapping: Dict[str, str] = {}

    if custom_groupings is None or len(custom_groupings) <= 0:
        for g in available_groups:
            group_mapping[g] = g

    else:
        for supergroup in custom_groupings:
            subgroups = supergroup.split(',')
            for subg in subgroups:
                if subg not in available_groups:
                    print(f'WARNING: subgroup "{subg}" from supergroup "{supergroup}" not found in model! Omitting...')
                    continue

                if subg in group_mapping:
                    print(f'WARNING: subgroup "{subg}" from supergroup "{supergroup}" already registered to supergroup "{group_mapping[subg]}"! Omitting...')
                    continue

                group_mapping[subg] = supergroup

    return group_mapping

def get_usage_representation(sentences: List[List[str]], max_syllable: int, bad_syllables: List[int]=[-5]):
    U = []
    for sentence in sentences:
        sentence = np.array([int(s) for s in sentence if s not in bad_syllables])
        u, _ = get_syllable_statistics([sentence], max_syllable=max_syllable, count='usage')
        u_vals = list(u.values())[:max_syllable]
        total_u = np.sum(u_vals)
        U.append(np.array(u_vals) / total_u)
    return np.array(U)

def get_transition_representation(sentences: List[List[str]], num_transitions: int, max_syllable: int,bad_syllables: List[int]=[-5]):

    tm_vals = []
    for sentence in sentences:
        sentence = np.array([int(s) for s in sentence if s not in bad_syllables])
        tm = get_transition_matrix([sentence], combine=True, max_syllable=max_syllable)
        tm_vals.append(tm.ravel())

    # Post-processing including truncation of transitions
    # Truncated transitions

    tm_vals_array = np.array(tm_vals)
    sorted_inds = np.argsort(tm_vals_array.mean(0))
    sorted_tm_vals = tm_vals_array[:,sorted_inds]
    if num_transitions < 0:
        tm_sums = list(sorted_tm_vals.sum(0))
        first_zero = max_syllable - next((i for i, x in enumerate(tm_sums) if x), None)
        T = sorted_tm_vals[:,:first_zero]
    else:
        T = sorted_tm_vals[:,-1*num_transitions:]
    return T

def get_embedding_representation(sentences: List[List[str]], emissions: bool, bad_syllables: List[int], dm: Literal[0,1,2], embedding_dim: int, embedding_window: int, embedding_epochs: int, min_count: int, negative: int, model_dest: str, ablation: str, phrase_path: str=None, seed=0):
     doc_embedding = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count, negative=negative, seed=seed)
     E = np.array(doc_embedding.fit_predict(sentences))
     doc_embedding.save(model_dest)
     return E
    
def fit_markov_chain(syllables, k, max_syllable=100):
    '''sample_markov_chain: using transition matrix `tmx`, sample from a markov chain `num_syllables` times'''

    syllable_array = np.array([int(s) for s in syllables])
    u, _ = get_syllable_statistics(syllable_array, max_syllable=max_syllable, count='usage')
    u_vals = list(u.values())
    total_u = np.sum(u_vals)
    usages = np.array(u_vals) / total_u

    u_dict = {str(i):usages[i] for i in range(max_syllable)}
    dist = DiscreteDistribution(u_dict)
    mc = MarkovChain([dist])
    return mc.from_samples(''.join(syllables),k)

def score_phrases(foreground_seqs, background_seqs, foreground_bigrams, min_count):

    '''score_phrases: assigns a score to each bigram in the foreground sequences based on Mikilov et al., 2013

        Positional args:
            foreground_seqs (list): a list of sequences from which significant phrases are extracted
            background_seqs (list): a list opf sequences against which the foreground sequences are compared for discriminating phrases
            foreground_bigrams (list): a list of precomputed bigrams in the foreground sequences
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
    '''
    
    # Unique elements in foreground seqs
    unique_foreground_els = []
    for el in foreground_seqs:
        if el not in unique_foreground_els:
            unique_foreground_els.append(el)

    len_vocab = len(unique_foreground_els)
    print(len_vocab)

    scored_bigrams = {}
    for a in tqdm(unique_foreground_els):
        for b in unique_foreground_els:
            if a == b: continue
            else: 
                count_a = background_seqs.count(a)
                count_b = background_seqs.count(b)

                bigram = f'{a}>{b}'
                count_ab = foreground_bigrams.count(bigram)
		# score = (#(ab in fg) - min) * len_vocab / #(a in bg)*#(b in bg)
                score = original_scorer(count_a, count_b, count_ab, len_vocab, min_count,-1)
                scored_bigrams[bigram] = score

    return scored_bigrams

def make_phrases(foreground_seqs, background_seqs, threshes, n, min_count):

    '''make_phrases: makes a dictionary containing disciminating phrases for a given class

        Positional args:
            foreground_seqs (list): a list of sequences from which significant phrases are extracted
            background_seqs (list): a list opf sequences against which the foreground sequences are compared for discriminating phrases
            threshes (list): a list of floating point thresholds which determine which n-gram phrases will be significant for each n
            n (int): number of times to run the agglomeration algorithm. Running n times will potentially yield up to 2n-grams
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
    '''

    # Flatten list of sequences into one long sequence (introduces artifacts?)
    flat_foreground_seqs = [el for seq in foreground_seqs for el in seq]
    flat_background_seqs = [el for seq in background_seqs for el in seq]

    all_phrases = {}

    # Calculate 2*n grams

    count = 0
    num_syl = float(len(flat_foreground_seqs))
    for m in range(n):

        # All non-unique bigrams in background and foreground sequences
        background_bigrams = [flat_background_seqs[i] + '>' + flat_background_seqs[i+1] for i in range(len(flat_background_seqs) - 1)]
        foreground_bigrams = [flat_foreground_seqs[i] + '>' + flat_foreground_seqs[i+1] for i in range(len(flat_foreground_seqs) - 1)]

        # Score bigrams in the foreground sequences
        print(f'Scoring bigrams: {m}')
        scored_bigrams = score_phrases(flat_foreground_seqs, flat_background_seqs, foreground_bigrams, min_count)

        # Threshold detected bigrams and replace in both the background and foreground sequences
        print('Thresholding...')
        #TODO: WARNING: This process will potentially eliminate some neighboring ngrams
        thresh = threshes[m]
        for bigram, score in tqdm(scored_bigrams.items()):
            ngram_len = len(bigram.split('>'))
            if score > thresh:
                all_phrases[bigram] = score
                while bigram in foreground_bigrams:
                    count += ngram_len
                    ind = foreground_bigrams.index(bigram)
                    del flat_foreground_seqs[ind]
                    del flat_foreground_seqs[ind]
                    del foreground_bigrams[ind]
                    flat_foreground_seqs.insert(ind, bigram)
                while bigram in background_bigrams:
                    ind = background_bigrams.index(bigram)
                    del flat_background_seqs[ind]
                    del flat_background_seqs[ind]
                    del background_bigrams[ind]
                    flat_background_seqs.insert(ind, bigram)
    prop = count / num_syl
    return (all_phrases, prop)

def make_phrases_dataset(sentences, labels, save_path, threshes, n, min_count):

    '''make_phrases_dataset: makes a dictionary containing disciminating phrases for each class in a dataset

        Positional args:
            sentences (List of strs): list of sentences representing moseq emissions
            labels (List of strs): list of class labels for each animal
            save_path (str): path for saving pickled dictionary of phrases
            threshes (list): a list of floating point thresholds which determine which n-gram phrases will be significant for each n
            n (int): number of times to run the agglomeration algorithm. Running n times will potentially yield up to 2n-grams
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
            '''
    # Get labels names
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    num_classes = len(unique_labels) 
    all_group_phrases = {}

    # For each group
    for label in unique_labels:

        # Compare label to other labels (including itself)
        foreground_sents = [seq for s, seq in enumerate(sentences) if labels[s] == label]
        background_sents = sentences
        all_group_phrases[labels] = make_phrases(foreground_sents, background_sents, threshes, n, min_count)

    # Save
    with open(save_path, 'wb') as handle:
        print(f'Saving at {handle}')
        pickle.dump(all_group_phrases, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
