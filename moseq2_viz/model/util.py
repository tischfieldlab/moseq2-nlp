from collections import defaultdict, OrderedDict
from copy import deepcopy
from sklearn.cluster import KMeans
from moseq2_viz.util import h5_to_dict
import numpy as np
import h5py
import pandas as pd
import warnings
import tqdm
import joblib
import os


def _get_transitions(label_sequence, fill_value=-5):

    # to_rem = np.where(label_sequence == fill_value)[0]
    arr = deepcopy(label_sequence)
    # arr = np.delete(arr, to_rem)
    # arr = np.insert(arr, len(arr), -10)
    # arr = np.insert(arr, 0, -10)

    locs = np.where(arr[1:] != arr[:-1])[0] + 1
    transitions = arr[locs]
    return transitions, locs


def _whiten_all(pca_scores, center=True):

    valid_scores = np.concatenate([x[~np.isnan(x).any(axis=1), :] for x in pca_scores.values()])
    mu, cov = valid_scores.mean(axis=0), np.cov(valid_scores, rowvar=False, bias=1)

    L = np.linalg.cholesky(cov)

    if center:
        offset = 0
    else:
        offset = mu

    whitened_scores = deepcopy(pca_scores)

    for k, v in whitened_scores.items():
        whitened_scores[k] = np.linalg.solve(L, (v - mu).T).T + offset

    return whitened_scores


# per https://gist.github.com/tg12/d7efa579ceee4afbeaec97eb442a6b72
def get_transition_matrix(labels, max_syllable=100, normalize='bigram',
                          smoothing=0.0, combine=False, disable_output=False):
    """Compute the transition matrix from a set of model labels

    Args:
        labels (list of np.array of ints): labels loaded from a model fit
        max_syllable (int): maximum syllable number to consider
        normalize (str): how to normalize transition matrix, 'bigram' or 'rows' or 'columns'
        smoothing (float): constant to add to transition_matrix pre-normalization to smooth counts
        combine (bool): compute a separate transition matrix for each element (False) or combine across all arrays in the list (True)

    Returns:
        transition_matrix (list): list of 2d np.arrays that represent the transitions from syllable i (row) to syllable j (column)

    Examples:

        Load in model results and get the transition matrix combined across sessions.

        >>> from moseq2_viz.model.util import parse_model_results, get_transition_matrix
        >>> model_results = parse_model_results('mymodel.p')
        >>> transition_matrix = get_transition_matrix(model_results['labels'], combine=True)

    """

    if combine:
        init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing

        for v in labels:

            transitions = _get_transitions(v)[0]

            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

        if normalize == 'bigram':
            init_matrix /= init_matrix.sum()
        elif normalize == 'rows':
            init_matrix /= init_matrix.sum(axis=1, keepdims=True)
        elif normalize == 'columns':
            init_matrix /= init_matrix.sum(axis=0, keepdims=True)
        else:
            pass

        all_mats = init_matrix
    else:

        all_mats = []
        for v in tqdm.tqdm(labels, disable=disable_output):

            init_matrix = np.zeros((max_syllable + 1, max_syllable + 1), dtype='float32') + smoothing
            transitions = _get_transitions(v)[0]

            for (i, j) in zip(transitions, transitions[1:]):
                if i <= max_syllable and j <= max_syllable:
                    init_matrix[i, j] += 1

            if normalize == 'bigram':
                init_matrix /= init_matrix.sum()
            elif normalize == 'rows':
                init_matrix /= init_matrix.sum(axis=1, keepdims=True)
            elif normalize == 'columns':
                init_matrix /= init_matrix.sum(axis=0, keepdims=True)
            else:
                pass

            all_mats.append(init_matrix)

    return all_mats


# return tuples with uuid and syllable indices
def get_syllable_slices(syllable, labels, label_uuids, index, trim_nans=True):

    h5s = [v['path'][0] for v in index['files'].values()]
    h5_uuids = list(index['files'].keys())

    # only extract if we have a match in the index
    # label_uuids = [uuid for uuid in label_uuids if uuid in h5_uuids]

    # grab the original indices from the pca file as well...

    if trim_nans:
        with h5py.File(index['pca_path'], 'r') as f:
            score_idx = h5_to_dict(f, 'scores_idx')

    sorted_h5s = [h5s[h5_uuids.index(uuid)] for uuid in label_uuids]
    syllable_slices = []

    for label_arr, label_uuid, h5 in zip(labels, label_uuids, sorted_h5s):

        if trim_nans:
            idx = score_idx[label_uuid]

            if len(idx) > len(label_arr):
                warnings.warn('Index length {:d} and label array length {:d} in {}'
                              .format(len(idx), len(label_arr), h5))
                idx = idx[:len(label_arr)]
            elif len(idx) < len(label_arr):
                warnings.warn('Index length {:d} and label array length {:d} in {}'
                              .format(len(idx), len(label_arr), h5))
                continue

            missing_frames = np.where(np.isnan(idx))[0]
            trim_idx = idx[~np.isnan(idx)].astype('int32')
            label_arr = label_arr[~np.isnan(idx)]
        else:
            missing_frames = None
            trim_idx = np.arange(len(label_arr))

        # do we need the trim_idx here actually?
        match_idx = trim_idx[np.where(label_arr == syllable)[0]]
        breakpoints = np.where(np.diff(match_idx, axis=0) > 1)[0]

        if len(breakpoints) < 1:
            continue

        breakpoints = zip(np.r_[0, breakpoints+1], np.r_[breakpoints, len(match_idx)-1])
        for i, j in breakpoints:
            # strike out movies that have missing frames
            if missing_frames is not None:
                if np.any(np.logical_and(missing_frames >= i, missing_frames <= j)):
                    continue
            syllable_slices.append([(match_idx[i], match_idx[j] + 1), label_uuid, h5])

    return syllable_slices


def get_syllable_statistics(data, fill_value=-5, max_syllable=100, count='usage'):
    """Compute the syllable statistics from a set of model labels

    Args:
        data (list of np.array of ints): labels loaded from a model fit
        max_syllable (int): maximum syllable to consider
        count (str): how to count syllable usage, either by number of emissions (usage), or number of frames (frames)

    Returns:
        usages (defaultdict): default dictionary of usages
        durations (defaultdict): default dictionary of durations

    Examples:

        Load in model results and get the transition matrix combined across sessions.

        >>> from moseq2_viz.model.util import parse_model_results, get_syllable_statistics
        >>> model_results = parse_model_results('mymodel.p')
        >>> usages, durations = get_syllable_statistics(model_results['labels'])

    """

    # if type(data) is list and type(data[0]) is np.ndarray:
    #     data = np.array([np.squeeze(tmp) for tmp in data], dtype='object')

    usages = defaultdict(int)
    durations = defaultdict(list)

    if count == 'usage':
        use_usage = True
    elif count == 'frames':
        use_usage = False
    else:
        raise RuntimeError('Did not understand count argument (must by usage or frames)')

    for s in range(max_syllable):
        usages[s] = 0
        durations[s] = []

    if type(data) is list or (type(data) is np.ndarray and data.dtype == np.object):

        for v in data:
            seq_array, locs = _get_transitions(v)
            to_rem = np.where(np.logical_or(seq_array > max_syllable,
                                            seq_array == fill_value))

            seq_array = np.delete(seq_array, to_rem)
            locs = np.delete(locs, to_rem)
            durs = np.diff(np.insert(locs, len(locs), len(v)))

            for s, d in zip(seq_array, durs):
                if use_usage:
                    usages[s] += 1
                else:
                    usages[s] += d
                durations[s].append(d)

    elif type(data) is np.ndarray and data.dtype == 'int16':

        seq_array, locs = _get_transitions(data)
        to_rem = np.where(seq_array > max_syllable)[0]

        seq_array = np.delete(seq_array, to_rem)
        locs = np.delete(locs, to_rem)
        durs = np.diff(np.insert(locs, len(locs), len(data)))

        for s, d in zip(seq_array, durs):
            if use_usage:
                usages[s] += 1
            else:
                usages[s] += d
            durations[s].append(d)

    usages = OrderedDict(sorted(usages.items()))
    durations = OrderedDict(sorted(durations.items()))

    return usages, durations


def labels_to_changepoints(labels, fs=30.):
    """Compute the transition matrix from a set of model labels

    Args:
        labels (list of np.array of ints): labels loaded from a model fit
        fs (float): sampling rate of camera

    Returns:
        cp_dist (list of np.array of floats): list of block durations per element in labels list

    Examples:

        Load in model results and get the changepoint distribution

        >>> from moseq2_viz.model.util import parse_model_results, labels_to_changepoints
        >>> model_results = parse_model_results('mymodel.p')
        >>> cp_dist = labels_to_changepoints(model_results['labels'])

    """

    cp_dist = []

    for lab in labels:
        cp_dist.append(np.diff(_get_transitions(lab)[1].squeeze()) / fs)

    return np.concatenate(cp_dist)


def parse_batch_modeling(filename):

    with h5py.File(filename, 'r') as f:
        results_dict = {
            'heldouts': np.squeeze(f['metadata/heldout_ll'].value),
            'parameters': h5_to_dict(f, 'metadata/parameters'),
            'scans': h5_to_dict(f, 'scans'),
            'filenames': [os.path.join(os.path.dirname(filename), os.path.basename(fname).decode('utf-8'))
                          for fname in f['filenames'].value],
            'labels': np.squeeze(f['labels'].value),
            'loglikes': np.squeeze(f['metadata/loglikes'].value),
            'label_uuids': [str(_, 'utf-8') for _ in f['/metadata/train_list'].value]
        }
        results_dict['scan_parameters'] = {k: results_dict['parameters'][k]
                                           for k in results_dict['scans'].keys() if k in results_dict['parameters'].keys()}

    return results_dict


def parse_model_results(model_obj, restart_idx=0, resample_idx=-1,
                        map_uuid_to_keys=False,
                        sort_labels_by_usage=False,
                        count='usage'):
    """Parses a model fit and returns a dictionary of results

    Args:
        model_obj (str or results returned from joblib.load): path to the model fit or a loaded model fit
        map_uuid_to_keys (bool): for labels, make a dictionary where each key, value pair contains the uuid and the labels for that session
        sort_labels_by_usage (bool): sort labels by their usages
        count (str): how to count syllable usage, either by number of emissions (usage), or number of frames (frames)

    Returns:
        output_dict (dict): dictionary with labels and model parameters

    Examples:

        Load in model results

        >>> from moseq2_viz.model.util import parse_model_results, labels_to_changepoints
        >>> model_results = parse_model_results('mymodel.p')

    """
    # reformat labels into something useful

    if type(model_obj) is str and (model_obj.endswith('.p') or model_obj.endswith('.pz')):
        model_obj = joblib.load(model_obj)
    elif type(model_obj) is str:
        raise RuntimeError('Can only parse models saved using joblib that end with .p or .pz')

    output_dict = deepcopy(model_obj)
    if type(output_dict['labels']) is list and type(output_dict['labels'][0]) is list:
        if np.ndim(output_dict['labels'][0][0]) == 2:
            output_dict['labels'] = [np.squeeze(tmp[resample_idx]) for tmp in output_dict['labels'][restart_idx]]
        elif np.ndim(output_dict['labels'][0][0]) == 1:
            output_dict['labels'] = [np.squeeze(tmp) for tmp in output_dict['labels'][restart_idx]]
        else:
            raise RuntimeError('Could not parse model labels')

    if type(output_dict['model_parameters']) is list:
        output_dict['model_parameters'] = output_dict['model_parameters'][restart_idx]

    if sort_labels_by_usage:
        output_dict['labels'], sorting = relabel_by_usage(output_dict['labels'], count=count)
        old_ar_mat = deepcopy(output_dict['model_parameters']['ar_mat'])
        old_nu = deepcopy(output_dict['model_parameters']['nu'])
        for i, sort_idx in enumerate(sorting):
            output_dict['model_parameters']['ar_mat'][i] = old_ar_mat[sort_idx]
            if type(output_dict['model_parameters']['nu']) is list:
                output_dict['model_parameters']['nu'][i] = old_nu[sort_idx]


    if map_uuid_to_keys:
        if 'train_list' in output_dict.keys():
            label_uuids = output_dict['train_list']
        else:
            label_uuids = output_dict['keys']

        label_dict = {uuid: lbl for uuid, lbl in zip(label_uuids, output_dict['labels'])}
        output_dict['labels'] = label_dict

    return output_dict


def relabel_by_usage(labels, fill_value=-5, count='usage'):
    """Resort model labels by their usages

    Args:
        labels (list of np.array of ints): labels loaded from a model fit
        fill_value (int): value prepended to modeling results to account for nlags
        count (str): how to count syllable usage, either by number of emissions (usage), or number of frames (frames)

    Returns:
        labels (list of np.array of ints): labels resorted by usage

    Examples:

        Load in model results and sort labels by usages

        >>> from moseq2_viz.model.util import parse_model_results, relabel_by_usage
        >>> model_results = parse_model_results('mymodel.p')
        >>> sorted_labels = relabel_by_usage(model_results['labels'])

    """

    sorted_labels = deepcopy(labels)
    usages, durations = get_syllable_statistics(labels, fill_value=fill_value, count=count)
    sorting = []

    for w in sorted(usages, key=usages.get, reverse=True):
        sorting.append(w)

    for i, v in enumerate(labels):
        for j, idx in enumerate(sorting):
            sorted_labels[i][np.where(v == idx)] = j

    return sorted_labels, sorting


def results_to_dataframe(model_dict, index_dict, sort=False, count='usage', normalize=True, max_syllable=40,
                         include_meta=['SessionName', 'SubjectName', 'StartTime']):

    if type(model_dict) is str:
        model_dict = parse_model_results(model_dict)

    if sort:
        model_dict['labels'] = relabel_by_usage(model_dict['labels'], count=count)[0]

    # by default the keys are the uuids

    if 'train_list' in model_dict.keys():
        label_uuids = model_dict['train_list']
    else:
        label_uuids = model_dict['keys']

    # durations = []

    df_dict = {
            'usage': [],
            'group': [],
            'syllable': []
        }

    for key in include_meta:
        df_dict[key] = []

    groups = [index_dict['files'][uuid]['group'] for uuid in label_uuids]
    metadata = [index_dict['files'][uuid]['metadata'] for uuid in label_uuids]

    for i, label_arr in enumerate(model_dict['labels']):
        tmp_usages, tmp_durations = get_syllable_statistics(label_arr, count=count, max_syllable=max_syllable)
        total_usage = np.sum(list(tmp_usages.values()))

        for k, v in tmp_usages.items():
            df_dict['usage'].append(v / total_usage)
            df_dict['syllable'].append(k)
            df_dict['group'].append(groups[i])

            for meta_key in include_meta:
                df_dict[meta_key].append(metadata[i][meta_key])

    df = pd.DataFrame.from_dict(data=df_dict)

    return df, df_dict


def simulate_ar_trajectory(ar_mat, init_points=None, sim_points=100):

    npcs = ar_mat.shape[0]

    if ar_mat.shape[1] % npcs == 1:
        affine_term = ar_mat[:, -1]
        ar_mat = np.delete(ar_mat, ar_mat.shape[1] - 1, axis=1)
    else:
        affine_term = np.zeros((ar_mat.shape[0], ), dtype='float32')

    nlags = ar_mat.shape[1] // npcs

    # print('Found {} pcs and {} lags in AR matrix'.format(npcs, nlags))

    if init_points is None:
        init_points = np.zeros((nlags, npcs), dtype='float32')

    sim_mat = np.zeros((sim_points + nlags, npcs), dtype='float32')
    sim_mat[:nlags] = init_points[:nlags]

    use_mat = np.zeros((nlags, npcs, npcs))

    for i in range(len(use_mat)):
        use_mat[i] = ar_mat[:, i * npcs: (i + 1) * npcs]

    for i in range(sim_points):
        sim_idx = i + nlags
        result = 0
        for j in range(1, nlags + 1):
            result += sim_mat[sim_idx - j].dot(use_mat[nlags - j])
        result += affine_term

        sim_mat[sim_idx, :] = result

    return sim_mat[nlags:]


def sort_batch_results(data, averaging=True, filenames=None, **kwargs):

    parameters = np.hstack(kwargs.values())
    param_sets = np.unique(parameters, axis=0)
    param_dict = {k: np.unique(v[np.isfinite(v)]) for k, v in kwargs.items()}

    param_list = list(param_dict.values())
    param_list = [p[np.isfinite(p)] for p in param_list]
    new_shape = tuple([len(v) for v in param_list])

    if filenames is not None:
        filename_index = np.empty(new_shape, dtype=np.object)
        for i, v in np.ndenumerate(filename_index):
            filename_index[i] = []
    else:
        filename_index = None

    dims = len(new_shape)

    if dims > 2:
        raise NotImplementedError('No support for more than 2 dimensions')

    if averaging:
        new_matrix = np.zeros(new_shape, dtype=data.dtype)
        new_count = np.zeros(new_shape, dtype=data.dtype)
    else:
        _, cnts = np.unique(parameters, return_counts=True, axis=0)
        nrestarts = cnts.max()
        if nrestarts == 0:
            raise RuntimeError('Did not detect any restarts')

        new_shape = tuple([nrestarts]) + new_shape
        new_matrix = np.zeros(new_shape, dtype=data.dtype)
        new_matrix[:] = np.nan

    # TODO: add support for no averaging (just default_dict or list)

    for param in param_sets:
        row_matches = np.where((parameters == param).all(axis=1))[0]
        idx = np.zeros((len(param),), dtype='int')

        if np.any(np.isnan(param)):
            continue

        for i, p in enumerate(param):
            idx[i] = int(np.where(param_list[i] == p)[0])

        for i, row in enumerate(row_matches):
            if dims == 2:
                if idx[0] >= 0 and idx[1] >= 0:
                    if averaging:
                        new_matrix[idx[0], idx[1]] = np.nansum([new_matrix[idx[0], idx[1]], data[row]])
                        new_count[idx[0], idx[1]] += 1
                    else:
                        new_matrix[i, idx[0], idx[1]] = data[row]
                    if filenames is not None:
                        filename_index[idx[0], idx[1]].append(filenames[row])
            elif dims == 1:
                if idx >= 0:
                    if averaging:
                        new_matrix[idx] = np.nansum([new_matrix[idx], data[row]])
                        new_count[idx] += 1
                    else:
                        new_matrix[i, idx] = data[row]
                    if filenames is not None:
                        filename_index[idx].append(filenames[row])

    if averaging:
        new_matrix[new_count == 0] = np.nan
        new_matrix /= new_count

    return new_matrix, param_dict, filename_index


def whiten_pcs(pca_scores, method='all', center=True):
    """Whiten PC scores using Cholesky whitening

    Args:
        pca_scores (dict): dictionary where values are pca_scores (2d np arrays)
        method (str): 'all' to whiten using the covariance estimated from all keys, or 'each' to whiten each separately
        center (bool): whether or not to center the data

    Returns:
        whitened_scores (dict): dictionary of whitened pc scores

    Examples:

        Load in pca_scores and whiten

        >>> from moseq2_viz.util import h5_to_dict
        >>> from moseq2_viz.model.util import whiten_pcs
        >>> pca_scores = h5_to_dict('pca_scores.h5', '/scores')
        >>> whitened_scores = whiten_pcs(pca_scores, method='all')

    """

    if method[0].lower() == 'a':
        whitened_scores = _whiten_all(pca_scores)
    else:
        whitened_scores = {}
        for k, v in pca_scores.items():
            whitened_scores[k] = _whiten_all({k: v})[k]

    return whitened_scores


def normalize_pcs(pca_scores, method='z'):
    """Normalize PCs (either de-mean or z-score)
    """

    norm_scores = deepcopy(pca_scores)
    if method.lower()[0] == 'z':
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        sig = np.nanstd(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = (v - mu) / sig
    elif method.lower()[0] == 'm':
        all_values = np.concatenate(list(norm_scores.values()), axis=0)
        mu = np.nanmean(all_values, axis=0)
        for k, v in norm_scores.items():
            norm_scores[k] = v - mu

    return norm_scores


def retrieve_pcs_from_slices(slices, pca_scores, max_dur=60,
                             max_samples=100, npcs=10, subsampling=None,
                             remove_offset=False, **kwargs):
    # pad using zeros, get dtw distances...

    durs = [idx[1] - idx[0] for idx, _, _ in slices]
    use_slices = [_ for i, _ in enumerate(slices) if durs[i] < max_dur]
    if max_samples is not None and len(use_slices) > max_samples:
        choose_samples = np.random.permutation(range(len(use_slices)))[:max_samples]
        use_slices = [_ for i, _ in enumerate(use_slices) if i in choose_samples]

    syllable_matrix = np.zeros((len(use_slices), max_dur, npcs), 'float32')
#     syllable_matrix[:] = np.nan

    for i, (idx, uuid, h5) in enumerate(use_slices):
        syllable_matrix[i, :idx[1]-idx[0], :] = pca_scores[uuid][idx[0]:idx[1], :npcs]

    if remove_offset:
        syllable_matrix = syllable_matrix - syllable_matrix[:, 0, :][:, None, :]

    if subsampling is not None and subsampling > 0:
        try:
            km = KMeans(subsampling)
            syllable_matrix = syllable_matrix.reshape(syllable_matrix.shape[0], max_dur * npcs)
            syllable_matrix = syllable_matrix[np.all(~np.isnan(syllable_matrix), axis=1), :]
            km.fit(syllable_matrix)
            syllable_matrix = km.cluster_centers_.reshape(subsampling, max_dur, npcs)
        except Exception:
            syllable_matrix = np.zeros((subsampling, max_dur, npcs))
            syllable_matrix[:] = np.nan

    return syllable_matrix
