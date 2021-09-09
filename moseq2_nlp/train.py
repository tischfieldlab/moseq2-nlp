from moseq2_nlp.utils import ensure_dir, write_yaml
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from moseq2_nlp.models import DocumentEmbedding
from moseq2_nlp.data import load_data


def train(name, save_dir, model_path, index_path, representation, emissions, custom_groupings, num_syllables, num_transitions, min_count, dm, embedding_dim, embedding_window,
          embedding_epochs, bad_syllables, scoring, K, penalty, num_c, seed):
    save_dict = {'parameters': locals()}
    times = {'Preamble': 0.0, 'Data': 0.0, 'Features': 0.0, 'Classifier': 0.0}
    start = time.time()

    if custom_groupings is not None:
        custom_groupings = [s.split(',') for s in custom_groupings]
    else:
        custom_groupings = []
    bad_syllables = [int(bs) for bs in bad_syllables]
    exp_dir = ensure_dir(os.path.join(save_dir, name))

    times['Preamble'] = time.time() - start

    start = time.time()
    print('Getting data')
    labels, usages, transitions, sentences, bigram_sentences = load_data(model_path,
                                                        index_path,
                                                        emissions=emissions,
                                                        custom_groupings=custom_groupings,
                                                        num_syllables=num_syllables,
                                                        num_transitions=num_transitions,
                                                        bad_syllables=bad_syllables)

    times['Data'] = time.time() - start

    start = time.time()
    print('Getting features')
    num_animals = len(labels)
    if representation == 'embeddings':
        model  = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count)
        rep = np.array(model.fit_predict(sentences))
        model.save(os.path.join(exp_dir, 'doc2vec'))
    elif representation == 'usages':
        rep = usages
    elif representation == 'transitions':
        rep = transitions
    else:
        raise ValueError('Representation type not recognized. Valid values are "usages", "transitions" and "embeddings".')
    times['Features'] = time.time() - start

    start = time.time()
    print('Training classifier')
    Cs = np.logspace(-5, 5, num_c)
    kf = KFold(n_splits=int(num_animals / float(K)))
    # Load and train classifier
    if penalty != 'none':
        clf = LogisticRegressionCV(Cs=Cs, cv=kf, scoring=scoring,random_state=seed, dual=False, solver='lbfgs', penalty=penalty,class_weight='balanced',multi_class='auto', tol=1e-6, max_iter=2000).fit(rep,labels)
    else:
        clf = LogisticRegressionCV(cv=kf, scoring=scoring,random_state=seed, dual=False, solver='lbfgs', class_weight='balanced', multi_class='auto', tol=1e-6, max_iter=2000).fit(rep,labels)
    scores = np.array([sc for sc in clf.scores_.values()]) # nm_classes x num_folds x num_C
    best_score = np.max(scores.mean((0,1)))
    best_C     = Cs[np.argmax(scores.mean((0,1)))]
    times['Classifier'] = time.time() - start

    save_dict['model_performance'] = {
        f'best_{scoring}': float(best_score),
        'best_C': float(best_C)
    }
    print(f'Best {scoring}: {best_score}')
    print(f'Best C: {best_C}')

    save_dict['compute_times'] = times
    write_yaml(os.path.join(exp_dir, 'experiment_info.yaml'), save_dict)
    np.save(os.path.join(exp_dir, f'{scoring}.npy'), best_score)
    np.save(os.path.join(exp_dir, 'best_C.npy'), best_C)
