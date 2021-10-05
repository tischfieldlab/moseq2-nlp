import os
import time
from typing import List, Literal

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from moseq2_nlp.data import get_embedding_representation, get_transition_representation, get_usage_representation, load_groups
from moseq2_nlp.utils import ensure_dir, write_yaml

Representation = Literal['embeddings', 'usages', 'transitions']
Penalty = Literal['l1', 'l2', 'elasticnet']


def train(name: str, save_dir: str, model_path: str, index_path: str, representation: Representation, emissions: bool, custom_groupings: List[str],
          num_syllables: int, num_transitions: int, min_count: int, dm: Literal[0,1,2], embedding_dim: int, embedding_window: int,
          embedding_epochs: int, bad_syllables: List[int], scoring: str, K: int, penalty: Penalty, num_c: int, seed:int):

    save_dict = {'parameters': locals()}
    times = {'Preamble': 0.0, 'Data': 0.0, 'Features': 0.0, 'Classifier': 0.0}


    start = time.time()

    group_map = load_groups(index_path, custom_groupings)
    bad_syllables = [int(bs) for bs in bad_syllables]
    exp_dir = ensure_dir(os.path.join(save_dir, name))

    times['Preamble'] = time.time() - start


    start = time.time()
    print('Getting features')
    if representation == 'embeddings':
        labels, features = get_embedding_representation(model_path, index_path, group_map, emissions=emissions, bad_syllables=bad_syllables,
                            dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count,
                            model_dest=os.path.join(exp_dir, 'doc2vec'))

    elif representation == 'usages':
        labels, features = get_usage_representation(model_path, index_path, group_map, num_syllables)

    elif representation == 'transitions':
        labels, features = get_transition_representation(model_path, index_path, group_map, num_transitions, max_syllable=num_syllables)

    else:
        raise ValueError('Representation type not recognized. Valid values are "usages", "transitions" and "embeddings".')

    times['Features'] = time.time() - start


    start = time.time()
    print('Training classifier')
    best_C, best_score = train_regressor(features, labels, K, scoring, penalty, num_c, seed)
    times['Classifier'] = time.time() - start


    print(f'Best {scoring}: {best_score}')
    print(f'Best C: {best_C}')

    save_dict['model_performance'] = {
        f'best_{scoring}': float(best_score),
        'best_C': float(best_C)
    }

    save_dict['compute_times'] = times
    write_yaml(os.path.join(exp_dir, 'experiment_info.yaml'), save_dict)

    np.save(os.path.join(exp_dir, f'{scoring}.npy'), best_score)
    np.save(os.path.join(exp_dir, 'best_C.npy'), best_C)



def train_regressor(features, labels, K: int, scoring: str, penalty: Penalty, num_c: int, seed: int):
    
    Cs = np.logspace(-5, 5, num_c)
    kf = KFold(n_splits=int(len(labels) / float(K)))

    params = {
        'cv': kf,
        'scoring': scoring,
        'random_state': seed,
        'dual': False,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
        'multi_class': 'auto',
        'tol': 1e-6,
        'max_iter': 2000
    }
    # Load and train classifier
    if penalty != 'none':
        params.update({
            'Cs': Cs,
            'penalty': penalty
        })

    clf = LogisticRegressionCV(**params).fit(features, labels)
    scores = np.array([sc for sc in clf.scores_.values()]) # nm_classes x num_folds x num_C
    best_score = np.max(scores.mean((0,1)))
    best_C     = Cs[np.argmax(scores.mean((0,1)))]

    return best_C, best_score
