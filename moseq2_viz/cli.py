from moseq2_viz.util import (recursive_find_h5s, check_video_parameters,
                             parse_index, h5_to_dict, clean_dict)
from moseq2_viz.model.util import (relabel_by_usage, get_syllable_slices,
                                   results_to_dataframe, parse_model_results,
                                   get_transition_matrix, get_syllable_statistics)
from moseq2_viz.viz import (make_crowd_matrix, usage_plot, graph_transition_matrix,
                            scalar_plot, position_plot)
from moseq2_viz.scalars.util import scalars_to_dataframe
from moseq2_viz.io.video import write_frames_preview
from functools import partial
from sys import platform
import click
import os
import ruamel.yaml as yaml
import h5py
import multiprocessing as mp
import numpy as np
import joblib
import tqdm
import warnings
import re
import shutil
import psutil

orig_init = click.core.Option.__init__


def new_init(self, *args, **kwargs):
    orig_init(self, *args, **kwargs)
    self.show_default = True


click.core.Option.__init__ = new_init


@click.group()
def cli():
    pass


@cli.command(name="add-group")
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--key', '-k', type=str, default='SubjectName', help='Key to search for value')
@click.option('--value', '-v', type=str, default='Mouse', help='Value to search for', multiple=True)
@click.option('--group', '-g', type=str, default='Group1', help='Group name to map to')
@click.option('--exact', '-e', type=bool, is_flag=True, help='Exact match only')
@click.option('--lowercase', type=bool, is_flag=True, help='Lowercase text filter')
@click.option('-n', '--negative', type=bool, is_flag=True, help='Negative match (everything that does not match is included)')
def add_group(index_file, key, value, group, exact, lowercase, negative):

    index = parse_index(index_file)[0]
    h5_uuids = [f['uuid'] for f in index['files']]
    metadata = [f['metadata'] for f in index['files']]

    if type(value) is str:
        value = [value]

    for v in value:
        if exact:
            v = r'\b{}\b'.format(v)
        if lowercase and negative:
            hits = [re.search(v, meta[key].lower()) is None for meta in metadata]
        elif lowercase:
            hits = [re.search(v, meta[key].lower()) is not None for meta in metadata]
        elif negative:
            hits = [re.search(v, meta[key]) is None for meta in metadata]
        else:
            hits = [re.search(v, meta[key]) is not None for meta in metadata]

        for uuid, hit in zip(h5_uuids, hits):
            position = h5_uuids.index(uuid)
            if hit:
                index['files'][position]['group'] = group

    new_index = '{}_update.yaml'.format(os.path.basename(index_file))

    try:
        with open(new_index, 'w+') as f:
            yaml.dump(index, f, Dumper=yaml.RoundTripDumper)
        shutil.move(new_index, index_file)
    except Exception:
        raise Exception


# recurse through directories, find h5 files with completed extractions, make a manifest
# and copy the contents to a new directory
@cli.command(name="copy-h5-metadata-to-yaml")
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--h5-metadata-path', default='/metadata/acquisition', type=str, help='Path to acquisition metadata in h5 files')
def copy_h5_metadata_to_yaml(input_dir, h5_metadata_path):

    h5s, dicts, yamls = recursive_find_h5s(input_dir)
    to_load = [(tmp, yml, file) for tmp, yml, file in zip(
        dicts, yamls, h5s) if tmp['complete'] and not tmp['skip']]

    # load in all of the h5 files, grab the extraction metadata, reformat to make nice 'n pretty
    # then stage the copy

    for i, tup in tqdm.tqdm(enumerate(to_load), total=len(to_load), desc='Copying data to yamls'):
        with h5py.File(tup[2], 'r') as f:
            tmp = clean_dict(h5_to_dict(f, h5_metadata_path))
            tup[0]['metadata'] = dict(tmp)

        try:
            new_file = '{}_update.yaml'.format(os.path.basename(tup[1]))
            with open(new_file, 'w+') as f:
                yaml.dump(tup[0], f, Dumper=yaml.RoundTripDumper)
            shutil.move(new_file, tup[1])
        except Exception:
            raise Exception


@cli.command(name='generate-index')
@click.option('--input-dir', '-i', type=click.Path(), default=os.getcwd(), help='Directory to find h5 files')
@click.option('--pca-file', '-p', type=click.Path(), default=os.path.join(os.getcwd(), '_pca/pca_scores.h5'), help='Path to PCA results')
@click.option('--output-file', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'moseq2-index.yaml'), help="Location for storing index")
@click.option('--filter', '-f', type=(str, str), default=None, help='Regex filter for metadata', multiple=True)
@click.option('--all-uuids', '-a', type=bool, default=False, help='Use all uuids')
def generate_index(input_dir, pca_file, output_file, filter, all_uuids):

    # gather than h5s and the pca scores file
    # uuids should match keys in the scores file

    h5s, dicts, yamls = recursive_find_h5s(input_dir)

    if not os.path.exists(pca_file) or all_uuids:
        warnings.warn('Will include all files')
        pca_uuids = [dct['uuid'] for dct in dicts]
    else:
        with h5py.File(pca_file, 'r') as f:
            pca_uuids = list(f['scores'].keys())

    file_with_uuids = [(os.path.relpath(h5), os.path.relpath(yml), meta) for h5, yml, meta in
                       zip(h5s, yamls, dicts) if meta['uuid'] in pca_uuids]

    if 'metadata' not in file_with_uuids[0][2]:
        raise RuntimeError('Metadata not present in yaml files, run copy-h5-metadata-to-yaml to update yaml files')

    output_dict = {
        'files': [],
        'pca_path': os.path.relpath(pca_file)
    }

    for i, file_tup in enumerate(file_with_uuids):
        output_dict['files'].append({
            'path': (file_tup[0], file_tup[1]),
            'uuid': file_tup[2]['uuid'],
            'group': 'default'
        })

        output_dict['files'][i]['metadata'] = {}

        for k, v in file_tup[2]['metadata'].items():
            for filt in filter:
                if k == filt[0]:
                    tmp = re.match(filt[1], v)
                    if tmp is not None:
                        v = tmp[0]

            output_dict['files'][i]['metadata'][k] = v

    # write out index yaml

    with open(output_file, 'w') as f:
        yaml.dump(output_dict, f, Dumper=yaml.RoundTripDumper)


@cli.command(name='make-crowd-movies')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-path', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('--max-examples', '-m', type=int, default=40, help="Number of examples to show")
@click.option('--threads', '-t', type=int, default=-1, help="Number of threads to use for rendering crowd movies")
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--output-dir', '-o', type=click.Path(), default=os.path.join(os.getcwd(), 'crowd_movies'), help="Path to store files")
#@click.option('--filename-format', type=str, default='syllable_{:d}.mp4', help="Python 3 string format for filenames")
@click.option('--gaussfilter-space', default=(0, 0), type=(float, float), help="Spatial filter for data (Gaussian)")
@click.option('--medfilter-space', default=[0], type=int, help="Median spatial filter", multiple=True)
@click.option('--min-height', type=int, default=5, help="Minimum height for scaling videos")
@click.option('--max-height', type=int, default=80, help="Minimum height for scaling videos")
@click.option('--raw-size', type=(int, int), default=(512, 424), help="Size of original videos")
@click.option('--scale', type=float, default=1, help="Scaling from pixel units to mm")
@click.option('--cmap', type=str, default='jet', help="Name of valid Matplotlib colormap for false-coloring images")
@click.option('--dur-clip', default=300, help="Exclude syllables more than this number of frames (None for no limit)")
@click.option('--legacy-jitter-fix', default=False, type=bool, help="Set to true if you notice jitter in your crowd movies")
def make_crowd_movies(index_file, model_path, max_syllable, max_examples,
                      threads, sort, count, gaussfilter_space, medfilter_space,
                      output_dir, min_height, max_height, raw_size, scale, cmap, dur_clip,
                      legacy_jitter_fix):

    if platform in ['linux', 'linux2']:
        print('Setting CPU affinity to use all CPUs...')
        cpu_count = psutil.cpu_count()
        proc = psutil.Process()
        proc.cpu_affinity(list(range(cpu_count)))

    clean_params = {
        'gaussfilter_space': gaussfilter_space,
        'medfilter_space': medfilter_space
    }

    # need to handle h5 intelligently here...

    if model_path.endswith('.p') or model_path.endswith('.pz'):
        model_fit = parse_model_results(joblib.load(model_path))
        labels = model_fit['labels']

        if 'train_list' in model_fit:
            label_uuids = model_fit['train_list']
        else:
            label_uuids = model_fit['keys']
    elif model_fit.endswith('.h5'):
        # load in h5, use index found using another function
        pass
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info_parameters = ['model_class', 'kappa', 'gamma', 'alpha']
    info_dict = {k: model_fit['model_parameters'][k] for k in info_parameters}

    # convert numpy dtypes to their corresponding primitives
    for k, v in info_dict.items():
        if isinstance(v, (np.ndarray, np.generic)):
            info_dict[k] = info_dict[k].item()

    info_dict['model_path'] = model_path
    info_dict['index_path'] = index_file
    info_file = os.path.join(output_dir, 'info.yaml')

    with open(info_file, 'w+') as f:
        yaml.dump(info_dict, f, Dumper=yaml.RoundTripDumper)

    if sort:
        labels, ordering = relabel_by_usage(labels, count=count)
    else:
        ordering = list(range(max_syllable))

    index, sorted_index = parse_index(index_file)
    vid_parameters = check_video_parameters(sorted_index)

    # uuid in both the labels and the index
    uuid_set = set(label_uuids) & set(sorted_index['files'].keys())

    # make sure the files exist
    uuid_set = [uuid for uuid in uuid_set if os.path.exists(sorted_index['files'][uuid]['path'][0])]

    # harmonize everything...
    labels = [label_arr for label_arr, uuid in zip(labels, label_uuids) if uuid in uuid_set]
    label_uuids = [uuid for uuid in label_uuids if uuid in uuid_set]
    sorted_index['files'] = {k: v for k, v in sorted_index['files'].items() if k in uuid_set}

    if vid_parameters['resolution'] is not None:
        raw_size = vid_parameters['resolution']

    if sort:
        filename_format = 'syllable_sorted-id-{:d} ({})_original-id-{:d}.mp4'
    else:
        filename_format = 'syllable_{:d}.mp4'

    with mp.Pool() as pool:
        slice_fun = partial(get_syllable_slices,
                            labels=labels,
                            label_uuids=label_uuids,
                            index=sorted_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            slices = list(tqdm.tqdm(pool.imap(slice_fun, range(max_syllable)), total=max_syllable))

        matrix_fun = partial(make_crowd_matrix,
                             nexamples=max_examples,
                             dur_clip=dur_clip,
                             min_height=min_height,
                             crop_size=vid_parameters['crop_size'],
                             raw_size=raw_size,
                             scale=scale,
                             legacy_jitter_fix=legacy_jitter_fix,
                             **clean_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", tqdm.TqdmSynchronisationWarning)
            crowd_matrices = list(tqdm.tqdm(pool.imap(matrix_fun, slices), total=max_syllable))

        write_fun = partial(write_frames_preview, fps=vid_parameters['fps'], depth_min=min_height,
                            depth_max=max_height, cmap=cmap)
        pool.starmap(write_fun,
                     [(os.path.join(output_dir, filename_format.format(i, count, ordering[i])),
                       crowd_matrix)
                      for i, crowd_matrix in enumerate(crowd_matrices) if crowd_matrix is not None])


@cli.command(name='plot-scalar-summary')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'scalars'))
def plot_scalar_summary(index_file, output_file):

    index, sorted_index = parse_index(index_file)
    scalar_df = scalars_to_dataframe(sorted_index)

    plt_scalars, _ = scalar_plot(scalar_df, headless=True)
    plt_position, _ = position_plot(scalar_df, headless=True)

    plt_scalars.savefig('{}_summary.png'.format(output_file))
    plt_scalars.savefig('{}_summary.pdf'.format(output_file))

    plt_position.savefig('{}_position.png'.format(output_file))
    plt_position.savefig('{}_position.pdf'.format(output_file))


@cli.command(name='plot-transition-graph')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('-g', '--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'transitions'), help="Filename to store plot")
@click.option('--normalize', type=click.Choice(['bigram', 'rows', 'columns']), default='bigram', help="How to normalize transition probabilities")
@click.option('--edge-threshold', type=float, default=.001, help="Threshold for edges to show")
@click.option('--usage-threshold', type=float, default=0, help="Threshold for nodes to show")
@click.option('--layout', type=str, default='spring', help="Default networkx layout algorithm")
@click.option('--keep-orphans', '-k', type=bool, is_flag=True, help="Show orphaned nodes")
@click.option('--orphan-weight', type=float, default=0, help="Weight for non-existent connections")
@click.option('--arrows', type=bool, is_flag=True, help="Show arrows")
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--edge-scaling', type=float, default=250, help="Scale factor from transition probabilities to edge width")
@click.option('--node-scaling', type=float, default=1e4, help="Scale factor for nodes by usage")
@click.option('--scale-node-by-usage', type=bool, default=True, help="Scale node sizes by usages probabilities")
@click.option('--width-per-group', type=float, default=8, help="Width (in inches) for figure canvas per group")
def plot_transition_graph(index_file, model_fit, max_syllable, group, output_file,
                          normalize, edge_threshold, usage_threshold, layout,
                          keep_orphans, orphan_weight, arrows, sort, count,
                          edge_scaling, node_scaling, scale_node_by_usage, width_per_group):

    if layout.lower()[:8] == 'graphviz':
        try:
            import pygraphviz
        except ImportError:
            raise ImportError('pygraphviz must be installed to use graphviz layout engines')

    model_data = parse_model_results(joblib.load(model_fit))
    index, sorted_index = parse_index(index_file)

    labels = model_data['labels']

    if sort:
        labels = relabel_by_usage(labels, count=count)[0]

    if 'train_list' in model_data.keys():
        label_uuids = model_data['train_list']
    else:
        label_uuids = model_data['keys']

    label_group = []

    print('Sorting labels...')

    if 'group' in index['files'][0].keys() and len(group) > 0:
        for uuid in label_uuids:
            label_group.append(sorted_index['files'][uuid]['group'])
    # elif 'group' in index['files'][0].keys() and (group is None or len(group) == 0):
    #     for uuid in label_uuids:
    #         label_group.append(sorted_index['files'][uuid]['group'])
    #     group = list(set(label_group))
    else:
        label_group = ['']*len(model_data['labels'])
        group = list(set(label_group))

    print('Computing transition matrices...')

    trans_mats = []
    usages = []
    for plt_group in group:
        use_labels = [lbl for lbl, grp in zip(labels, label_group) if grp == plt_group]
        trans_mats.append(get_transition_matrix(use_labels, normalize=normalize, combine=True, max_syllable=max_syllable))
        usages.append(get_syllable_statistics(use_labels)[0])

    if not scale_node_by_usage:
        usages = None

    print('Creating plot...')

    plt, _, _ = graph_transition_matrix(trans_mats, usages=usages, width_per_group=width_per_group,
                                        edge_threshold=edge_threshold, edge_width_scale=edge_scaling,
                                        difference_edge_width_scale=edge_scaling, keep_orphans=keep_orphans,
                                        orphan_weight=orphan_weight, arrows=arrows, usage_threshold=usage_threshold,
                                        layout=layout, groups=group, usage_scale=node_scaling, headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))


@cli.command(name='plot-usages')
@click.argument('index-file', type=click.Path(exists=True, resolve_path=True))
@click.argument('model-fit', type=click.Path(exists=True, resolve_path=True))
@click.option('--sort', type=bool, default=True, help="Sort syllables by usage")
@click.option('--count', type=click.Choice(['usage', 'frames']), default='usage', help='How to quantify syllable usage')
@click.option('--max-syllable', type=int, default=40, help="Index of max syllable to render")
@click.option('-g', '--group', type=str, default=None, help="Name of group(s) to show", multiple=True)
@click.option('--output-file', type=click.Path(), default=os.path.join(os.getcwd(), 'usages'), help="Filename to store plot")
def plot_usages(index_file, model_fit, sort, count, max_syllable, group, output_file):

    # if the user passes multiple groups, sort and plot against each other
    # relabel by usage across the whole dataset, gather usages per session per group

    # parse the index, parse the model fit, reformat to dataframe, bob's yer uncle

    model_data = parse_model_results(joblib.load(model_fit))
    index, sorted_index = parse_index(index_file)
    df, _ = results_to_dataframe(model_data, sorted_index, max_syllable=max_syllable, sort=sort, count=count)
    plt, _ = usage_plot(df, groups=group, headless=True)
    plt.savefig('{}.png'.format(output_file))
    plt.savefig('{}.pdf'.format(output_file))
