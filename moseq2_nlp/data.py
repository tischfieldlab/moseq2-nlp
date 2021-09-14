
from typing import List

import numpy as np
from gensim.models import Phrases
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from tqdm import tqdm


def load_data(model_file: str, index_file: str, emissions: bool, num_syllables: int, num_transitions: int,
              bad_syllables: List[int], custom_groupings: List[str]):
    ''' Load data from a moseq model

    Parameters:
        model_file (str): path to moseq model pickle file
        index_file (str): path to moseq model index yaml file
        ...
    '''

    _, sorted_index = parse_index(index_file)

    ms_model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    labels = ms_model['labels']
    label_group = [sorted_index['files'][uuid]['group'] for uuid in ms_model['keys']]

    # Get group names
    groups = []
    custom_labels = []
    for lg in label_group:
        if lg in groups:
            continue
        else:
            groups.append(lg)

    # Potentially use custom numerical labels
    if len(custom_groupings) > 0:
        for g in groups:
            for c, cg in enumerate(custom_groupings):
                if g in cg: 
                    custom_labels.append(c)
    else:
        custom_labels = [i for i in range(len(groups))]

    # Relabel according to custom groups
    if len(custom_groupings) > 0:
        custom_label_names = []
        for cg in custom_groupings:
            custom_label_names.append(','.join(cg))
    else:
        custom_label_names = groups

    tm_vals = []
    truncated_tm_vals = []
    group_vals = []
    group_labels = []
    usage_vals = []
    frames_vals = []
    sentences = []
    bigram_sentences = []
    sentence_strings = []
    sentence_groups = {group : [] for group in groups}
    for i, (l, g) in tqdm(enumerate(zip(labels, label_group))):

        if g not in groups and g != 'M_ukn':
            raise ValueError('Group name in data not recognized. Check the group names you specified!')
        elif g == 'M_ukn':
            continue

        group_vals.append(g)

        # Label data using default or custom labels
        group_labels.append(custom_labels[groups.index(g)])

        # Get transitions
        tm = get_transition_matrix([l], combine=True, max_syllable=num_syllables - 1)
        tm_vals.append(tm.ravel())

        # Get usages
        u, _ = get_syllable_statistics(l, count='usage')
        u_vals = list(u.values())[:num_syllables]
        total_u = np.sum(u_vals)
        usage_vals.append(np.array(u_vals) / total_u)

        # Get frame values
        f, _ = get_syllable_statistics(l, count='usage')
        total_f = np.sum(list(f.values()))
        frames_vals.append(np.array(list(f.values())) / total_f)

        # Get emissions
        l = list(filter(lambda a: a not in bad_syllables, l))
        np_l = np.array(l)
        if emissions:
            cp_inds = np.concatenate((np.where(np.diff(np_l) != 0 )[0],np.array([len(l) - 1])))
            syllables = np_l[cp_inds]
        else:
            syllables = np_l
        sentence = [str(syl) for syl in syllables]
        sentences.append(sentence)
        sentence_strings.append(' '.join(sentence))
        sentence_groups[g].append(sentence)

        bigram_model = Phrases(sentence, min_count=1, threshold=1, scoring='default')
        bgs = bigram_model[sentence]
        bigram_sentences.append(bgs)

    # Post-processing including truncation of transitions
    # Truncated transitions
    tm_vals = np.array(tm_vals)
    top_transitions = np.argsort(tm_vals.mean(0))[-num_transitions:]
    truncated_tm_vals = tm_vals[:,top_transitions]

    # Make numpy
    usage_vals = np.array(usage_vals)
    frames_vals = np.array(frames_vals)
    num_animals = len(sentences)

    np_g = np.array(group_labels)
    group_sizes = [sum(np_g == g) for g in np.unique(np_g)]
    lb_ind = np.argsort(np_g)
    return group_labels, usage_vals, truncated_tm_vals, sentences, bigram_sentences
