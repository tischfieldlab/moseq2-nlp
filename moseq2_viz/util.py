import os
import h5py
from ruamel.yaml import YAML
import numpy as np
import re


# https://gist.github.com/jaytaylor/3660565
_underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
_underscorer2 = re.compile('([a-z0-9])([A-Z])')


def camel_to_snake(s):
    """Converts CamelCase to snake_case
    """
    subbed = _underscorer1.sub(r'\1_\2', s)
    return _underscorer2.sub(r'\1_\2', subbed).lower()


def check_video_parameters(index):

    ymls = [v['path'][1] for v in index['files'].values()]

    dicts = []

    yaml = YAML(typ='safe')

    for yml in ymls:
        with open(yml, 'r') as f:
            dicts.append(yaml.load(f.read()))

    check_parameters = ['crop_size', 'fps', 'max_height', 'min_height']

    if 'resolution' in list(dicts[0]['parameters'].keys()):
        check_parameters.append('resolution')

    for chk in check_parameters:
        tmp_list = [dct['parameters'][chk] for dct in dicts]
        if not all(x == tmp_list[0] for x in tmp_list):
            raise RuntimeError('Parameter {} not equal in all extractions'.format(chk))

    vid_parameters = {
        'crop_size': tuple(dicts[0]['parameters']['crop_size']),
        'fps': dicts[0]['parameters']['fps'],
        'max_height': dicts[0]['parameters']['max_height'],
        'min_height': dicts[0]['parameters']['min_height'],
        'resolution': None
    }

    if 'resolution' in check_parameters:
        vid_parameters['resolution'] = tuple([tmp+100 for tmp in dicts[0]['parameters']['resolution']])

    return vid_parameters


# def commented_map_to_dict(cmap):
#
#     new_var = dict()
#
#     if type(cmap) is CommentedMap or type(cmap) is dict:
#         for k, v in cmap.items():
#             if type(v) is CommentedMap or type(v) is dict:
#                 new_var[k] = commented_map_to_dict(v)
#             elif type(v) is np.ndarray:
#                 new_var[k] = v.tolist()
#             elif isinstance(v, np.generic):
#                 new_var[k] = np.asscalar(v)
#             else:
#                 new_var[k] = v
#
#     return new_var


def clean_dict(dct):

    new_var = dict()

    if type(dct) is dict:
        for k, v in dct.items():
            if type(v) is dict:
                new_var[k] = clean_dict(v)
            elif type(v) is np.ndarray:
                new_var[k] = v.tolist()
            elif isinstance(v, np.generic):
                new_var[k] = np.asscalar(v)
            else:
                new_var[k] = v

    return new_var


def _load_h5_to_dict(file: h5py.File, path: str) -> dict:
    ans = {}
    for key, item in file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = _load_h5_to_dict(file, '/'.join([path, key]))
    return ans


def h5_to_dict(h5file, path: str) -> dict:
    '''
    Args:
        h5file (str or h5py.File): file path to the given h5 file or the h5 file handle
        path: path to the base dataset within the h5 file
    Returns:
        a dict with h5 file contents with the same path structure
    '''
    if isinstance(h5file, str):
        with h5py.File(h5file, 'r') as f:
            out = _load_h5_to_dict(f, path)
    elif isinstance(h5file, h5py.File):
        out = _load_h5_to_dict(h5file, path)
    else:
        raise Exception('file input not understood - need h5 file path or file object')
    return out


def load_changepoints(cpfile):
    with h5py.File(cpfile, 'r') as f:
        cps = h5_to_dict(f, 'cps')

    cp_dist = []

    for k, v in cps.items():
        cp_dist.append(np.diff(v.squeeze()))

    return np.concatenate(cp_dist)


def load_timestamps(timestamp_file, col=0):
    """Read timestamps from space delimited text file
    """

    ts = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            cols = line.split()
            ts.append(float(cols[col]))

    return np.array(ts)


def parse_index(index_file, get_metadata=False):

    yaml = YAML(typ='safe')

    with open(index_file, 'r') as f:
        index = yaml.load(f)

    # sort index by uuids

    # yaml_dir = os.path.dirname(index_file)

    index_dir = os.path.dirname(index_file)
    h5s = [(os.path.join(index_dir, idx['path'][0]),
            os.path.join(index_dir, idx['path'][1]))
           for idx in index['files']]
    h5_uuids = [idx['uuid'] for idx in index['files']]
    groups = [idx['group'] for idx in index['files']]
    metadata = [idx['metadata']
                if 'metadata' in idx.keys() else {} for idx in index['files']]

    sorted_index = {
        'files': {},
        'pca_path': os.path.join(index_dir, index['pca_path'])
    }

    for uuid, h5, group, h5_meta in zip(h5_uuids, h5s, groups, metadata):
        sorted_index['files'][uuid] = {
            'path':  h5,
            'group': group,
            'metadata': h5_meta
        }

    # ymls = ['{}.yaml'.format(os.path.splitext(h5)[0]) for h5 in h5s]

    return index, sorted_index


def recursive_find_h5s(root_dir=os.getcwd(),
                       ext='.h5',
                       yaml_string='{}.yaml'):
    """Recursively find h5 files, along with yaml files with the same basename
    """
    dicts = []
    h5s = []
    yamls = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            yaml_file = yaml_string.format(os.path.splitext(file)[0])
            if file.endswith(ext) and os.path.exists(os.path.join(root, yaml_file)):
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'frames' not in f.keys():
                        continue
                h5s.append(os.path.join(root, file))
                yamls.append(os.path.join(root, yaml_file))
                dicts.append(read_yaml(os.path.join(root, yaml_file)))

    return h5s, dicts, yamls


def read_yaml(yaml_file):

    yaml = YAML(typ='safe')

    with open(yaml_file, 'r') as f:
        dat = f.read()
        try:
            return_dict = yaml.load(dat)
        except yaml.constructor.ConstructorError:
            return_dict = yaml.load(dat)

    return return_dict


# from https://stackoverflow.com/questions/40084931/taking-subarrays-from-numpy-array-with-given-stride-stepsize/40085052#40085052
# dang this is fast!
def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S*n, n))
