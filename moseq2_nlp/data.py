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

def get_raw_data(model_file: str, index_file: str, max_syllable: int=100, emissions: bool=True, bad_syllables: List[int]=[-5], ablation: str='none', phrase_path: str=None):
    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    label_group = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]

    sentences = []
    out_groups: List[str] = []
    for l, g in zip(tqdm(model['labels']), label_group):
        l = list(filter(lambda a: a not in bad_syllables, l))
        np_l = np.array(l)
        if emissions:
            cp_inds = np.concatenate((np.where(np.diff(np_l) != 0 )[0],np.array([len(l) - 1])))
            syllables = np_l[cp_inds]
        else:
            syllables = np_l
        sentence = [str(syl) for syl in syllables]
        sentences.append(sentence)
        out_groups.append(g)

    if ablation != 'none':
        if emissions is False:
            raise ValueError('Ablation only works with emission data!')

        # Load phrases for each group
        if phrase_path is not None:
            with open(phrase_path, 'rb') as handle:
                group_dict = pickle.load(handle)

        else:
            raise ValueError('Expected a `phrase_path` str, but got `None`!')

        # For each group, phrase_dict and proportion of deleted syllables
        for g, (group, (phrase_dict, prop)) in enumerate(group_dict.items()):

            # For each sentence
            for s, sentence in enumerate(sentences):
                if out_groups[s] != group: continue
                else: 
                    if ablation == 'targeted':
                        # For each phrase
                        for phrase in phrase_dict.keys():
                            # Turn phrase into list
                            dissociated_phrase = phrase.split('>')
                            len_phrase = len(dissociated_phrase)
                            for m, module in enumerate(sentence):
                                # If phrase detected
                                if sentence[m:m + len_phrase] == dissociated_phrase:
                                    # Get random phrase without identical neighbors
                                    rand_phrase = np.zeros((len_phrase))
                                    while 0 in np.diff(rand_phrase):
                                        rand_phrase = np.random.randint(0,max_syllable, len_phrase)

                                    # Insert random phrase at target location
                                    sentences[s][m:m + len_phrase] = [str(el) for el in rand_phrase]

                    elif ablation == 'random':
                        for m, module in enumerate(sentence):
                            # Randomly replace syllables at rate prop
                            if np.random.rand() < prop:
                                sentences[s][m] = str(np.random.randint(0,max_syllable))
    return sentences, out_groups

def get_usage_representation(model_file: str, index_file: str, group_map: Dict[str, str], max_syllable: int=100):
    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    label_group = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]

    usage_vals = []
    out_groups = []
    for l, g in zip(tqdm(model['labels']), label_group):
        if g in group_map.keys():
            u, _ = get_syllable_statistics(l, max_syllable=max_syllable, count='usage')
            u_vals = list(u.values())
            total_u = np.sum(u_vals)
            usage_vals.append(np.array(u_vals) / total_u)
            out_groups.append(group_map[g])
    return out_groups, np.array(usage_vals)


def get_transition_representation(model_file: str, index_file: str, group_map: Dict[str, str], num_transitions: int, max_syllable: int=100):
    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    label_group = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]

    tm_vals = []
    out_groups = []
    for l, g in zip(tqdm(model['labels']), label_group):
        if g in group_map.keys():
            tm = get_transition_matrix([l], combine=True, max_syllable=max_syllable)
            tm_vals.append(tm.ravel())
            out_groups.append(group_map[g])

    # Post-processing including truncation of transitions
    # Truncated transitions

    tm_vals_array = np.array(tm_vals)
    sorted_inds = np.argsort(tm_vals_array.mean(0))
    sorted_tm_vals = tm_vals_array[:,sorted_inds]
    if num_transitions < 0:
       tm_sums = list(sorted_tm_vals.sum(0))
       first_zero = max_syllable - next((i for i, x in enumerate(tm_sums) if x), None)
       truncated_tm_vals = sorted_tm_vals[:,:first_zero]
    else:
        truncated_tm_vals = sorted_tm_vals[:,num_transitions]

    return out_groups, truncated_tm_vals

def get_embedding_representation(model_file: str, index_file: str, group_map: Dict[str, str], emissions: bool, bad_syllables: List[int], dm: Literal[0,1,2], embedding_dim: int, embedding_window: int, embedding_epochs: int, min_count: int, negative: int, model_dest: str, ablation: str, phrase_path: str=None, seed=0):

    sentences, out_groups = get_raw_data(model_file, index_file, max_syllable=100, emissions=emissions, bad_syllables=bad_syllables, ablation=ablation, phrase_path=phrase_path)

    doc_embedding = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count, negative=negative, seed=seed)
    rep = np.array(doc_embedding.fit_predict(sentences))
    doc_embedding.save(model_dest)

    return out_groups, rep
    
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

def make_phrases_dataset(model_path, index_path, save_path, threshes, n, min_count):

    '''make_phrases_dataset: makes a dictionary containing disciminating phrases for each class in a dataset

        Positional args:
            model_path (str): path of .p model file
            index_path (str): path of .yaml index file
            save_path (str): path for saving pickled dictionary of phrases
            threshes (list): a list of floating point thresholds which determine which n-gram phrases will be significant for each n
            n (int): number of times to run the agglomeration algorithm. Running n times will potentially yield up to 2n-grams
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
            '''
    # Load raw data
    sequences, out_groups = get_raw_data(model_path, index_path)

    # Get group names
    unique_groups = []
    for group in out_groups:
        if group not in unique_groups:
            unique_groups.append(group)
    num_groups = len(unique_groups) 
    all_group_phrases = {}

    # For each group
    for group in unique_groups:

        # Compare group to other groups (including itself)
        foreground_seqs = [seq for s, seq in enumerate(sequences) if out_groups[s] == group]
        background_seqs = sequences
        all_group_phrases[group] = make_phrases(foreground_seqs, background_seqs, threshes, n, min_count)

    # Save
    with open(save_path, 'wb') as handle:
        pickle.dump(all_group_phrases, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return all_group_phrases
