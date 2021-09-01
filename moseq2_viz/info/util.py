import numpy as np
from moseq2_viz.model.util import get_syllable_statistics, get_transition_matrix


def entropy(labels, truncate_syllable=40, smoothing=1.0):
    """Computes entropy, base 2
    """

    ent = []
    for v in labels:
        usages = get_syllable_statistics([v])[0]

        syllables = np.array(list(usages.keys()))
        truncate_point = np.where(syllables == truncate_syllable)[0]

        if truncate_point is None or len(truncate_point) != 1:
            truncate_point = len(syllables)
        else:
            truncate_point = truncate_point[0]

        syllables = syllables[:truncate_point]

        usages = np.array(list(usages.values())).astype('float')
        usages = usages[:truncate_point] + smoothing
        usages /= usages.sum()

        ent.append(-np.sum(usages * np.log2(usages)))

    return ent


def entropy_rate(labels, truncate_syllable=40, normalize='bigram',
                 smoothing=1.0, tm_smoothing=1.0):
    """Computes entropy rate, base 2
    """

    ent = []
    for v in labels:

        usages = get_syllable_statistics([v])[0]
        syllables = np.array(list(usages.keys()))
        truncate_point = np.where(syllables == truncate_syllable)[0]

        if truncate_point is None or len(truncate_point) != 1:
            truncate_point = len(syllables)
        else:
            truncate_point = truncate_point[0]

        syllables = syllables[:truncate_point]

        usages = np.array(list(usages.values())).astype('float')
        usages = usages[:truncate_point] + smoothing
        usages /= usages.sum()

        tm = get_transition_matrix([v],
                                   max_syllable=100,
                                   normalize='none',
                                   smoothing=0.0,

                                   disable_output=True)[0] + tm_smoothing
        tm = tm[:truncate_point]
        tm = tm[:, :truncate_point]

        if normalize == 'bigram':
            tm /= tm.sum()
        elif normalize == 'rows':
            tm /= tm.sum(axis=1, keepdims=True)
        elif normalize == 'columns':
            tm /= tm.sum(axis=0, keepdims=True)

        ent.append(-np.sum(usages[:, None] * tm * np.log2(tm)))

    return ent
