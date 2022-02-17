from typing import Dict, List, Literal

import numpy as np
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from gensim.models.phrases import original_scorer
from tqdm import tqdm
import pdb
from moseq2_nlp.models import DocumentEmbedding

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

def get_raw_data(model_file: str, index_file: str, max_syllable: int=100, emissions: bool=True, bad_syllables: List[int]=[-5]):
    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    label_group = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]

    sentences = []
    out_groups = []
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
    top_transitions = np.argsort(tm_vals_array.mean(0))[-num_transitions:]
    truncated_tm_vals = tm_vals_array[:,top_transitions]

    return out_groups, truncated_tm_vals


def get_embedding_representation(model_file: str, index_file: str, group_map: Dict[str, str], emissions: bool, bad_syllables: List[int], dm: Literal[0,1,2],
                                 embedding_dim: int, embedding_window: int, embedding_epochs: int, min_count: int, model_dest: str):

    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    label_group = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]

    sentences = []
    out_groups = []
    for l, g in zip(tqdm(model['labels']), label_group):
        if g in group_map.keys():
            l = list(filter(lambda a: a not in bad_syllables, l))
            np_l = np.array(l)
            if emissions:
                cp_inds = np.concatenate((np.where(np.diff(np_l) != 0 )[0],np.array([len(l) - 1])))
                syllables = np_l[cp_inds]
            else:
                syllables = np_l
            sentence = [str(syl) for syl in syllables]
            sentences.append(sentence)
            out_groups.append(group_map[g])

    doc_embedding = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count)
    rep = np.array(doc_embedding.fit_predict(sentences))
    doc_embedding.save(model_dest)

    return out_groups, rep

def score_phrases(foreground_seqs, background_seqs, foreground_bigrams, min_count):

    # Unique elements in foreground seqs
    unique_foreground_els = []
    for el in foreground_seqs:
        if el not in unique_foreground_els:
            unique_foreground_els.append(el)

    len_vocab = len(unique_foreground_els)

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

    # Flatten list of sequences into one long sequence (introduces artifacts?)
    flat_foreground_seqs = [el for seq in foreground_seqs for el in seq]
    flat_background_seqs = [el for seq in background_seqs for el in seq]

    all_phrases = {}

    # Calculate 2*n grams
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
            if score > thresh:
                all_phrases[bigram] = score
                while bigram in foreground_bigrams:
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
    return all_phrases
