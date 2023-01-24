from typing import Dict, List, Literal, Optional, Union
#from pomegranate import DiscreteDistribution, MarkovChain
import numpy as np
from moseq2_viz.model.util import (get_syllable_statistics,
                                   get_transition_matrix, parse_model_results)
from moseq2_viz.util import parse_index
from moseq2_nlp.util import get_unique_list_elements
from gensim.models.phrases import original_scorer
from tqdm import tqdm
import pdb
from moseq2_nlp.models import DocumentEmbedding
from scipy.sparse import coo_matrix
import pickle

def load_groups(index_file: str, custom_groupings: List[str]) -> Dict[str, str]:
    """Load data and group into classes according to custom_groupings, if provided.

    Args:
        index_file: yaml file indexing moseq data.
        custom_groupings: a list of strings; each string contains one or more comma-separated groups; each element of the list is grouped into a single class 

    Returns:
        A dictionary mapping from subgroup (key) to supergroup (value)
        and dtype `float32`.

    See also: `get_raw`
    """
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
    """Compute usage (bag of words) representations.

    Args:
        sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
        max_syllable: an int indicating the maximum value of syllables to be counted
        bad_syllables: a list, defaulting to [-5], indicating syllables to be excluded from the count. 

    Returns:
        U: The usage representation of all animals in the form of an animal x max_syllable np.float32 array; i.e. a normalized histogram of syllable counts.
           Syllables are sorted by frequency

    See also: `get_raw, get transition_representation, get_embedding_representation`
    """
 
    U = []
    for sentence in sentences:
        sentence = np.array([int(s) for s in sentence if s not in bad_syllables])
        u, _ = get_syllable_statistics([sentence], max_syllable=max_syllable, count='usage')
        u_vals = list(u.values())[:max_syllable]
        total_u = np.sum(u_vals)
        U.append(np.array(u_vals) / total_u)
    U = np.array(U)
    return U

def get_transition_representation(sentences: List[List[str]], num_transitions: int, max_syllable: int,bad_syllables: List[int]=[-5]):
    """Compute transition (normalized bigram) representations.

    Args:
        sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
        num_transitions: an int indicating the number of transitions to include. If < 0, then num_transitions is set to that value after which transition probs are zero for all animals
        max_syllable: an int indicating the maximum value of syllables to be counted
        bad_syllables: a list, defaulting to [-5], indicating syllables to be excluded from the count. 

    Returns:
        T: The transition representation of all animals in the form of an animal x num_transitoin np.float32 array; i.e. truncated and flattened transition matrix,
        first normalized by row and column.

    See also: `get_raw, get usage_representation, get_embedding_representation`
    """
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

def get_embedding_representation(sentences: List[List[str]], bad_syllables: List[int], dm: Literal[0,1,2], embedding_dim: int, embedding_window: int, embedding_epochs: int, min_count: int, negative: int, model_dest: str, seed=0, return_syllable_embeddings=False):
     """Compute embedding (doc2vec) representations. See https://radimrehurek.com/gensim/models/doc2vec.html

     Args:
        sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
        bad_syllables: a list, defaulting to [-5], indicating syllables to be excluded from the count. 
        dm: literal with value 0, 1 or 2. 0 means dbow representation; 1 means distributed memory representation; 2 means an average of the two. 
        embedding_dim: int indicating the dimension of the latent space
        embedding_window: int controling the size of the window in which doc2vec predicts word context. 
        embedding_epochs: int indicating the number of training epohcs
        min_count: the minimum number of times a word must appear to be included in the vocabulary
        negative: int indicating the exponent for negative sampling
        model_est: str indicating where model will be saved
        seed: int for random seed
        return_syllable_embeddings: bool; if True, returns embeddings for individual syllables

     Returns:
         E: a num_animal x embedding_dim np.float32 array containing embedding representations for each animal

     See also: `get_raw, get usage_representation, get_transition_representation`
     """

     # TODO: MAX SYLLABLE?

     doc_embedding = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count, negative=negative, seed=seed)
     E = np.array(doc_embedding.fit_predict(sentences))
     doc_embedding.save(model_dest)
     return E

def get_emissions(sentences):
    """Convert a sequence of raw syllables to a sequence of emissions 

        Args:
            sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.

        Returns: 
            emissions: sentences in which raw syllables are replaced by emissions so frame (real-time) information is lost
     """
    emissions = []
    for sentence in sentences:
        sentence = np.array([int(s) for s in sentence])
        cps = np.concatenate([np.where(np.diff(sentence) != 0)[0], np.array([len(sentence) - 1])], axis=0)
        emissions.append([str(s) for s in sentence[cps]])
    return emissions

def get_raw(index_file: str, model_file: str, custom_groupings: List[str]):

    """Load raw syllables and labels from a moseq model

       Args:
           index_file: yaml file indexing moseq data.
           model_file: p file with raw syllables 
           custom_groupings: a list of strings; each string contains one or more comma-separated groups; each element of the list is grouped into a single class 

       Returns:
           sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
           labels: labels for each animal's syllables. Labels possibly consolidated into supergroups by custom labels

    """

    _, sorted_index = parse_index(index_file)
    model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    sentences = model['labels']
    raw_labels = [sorted_index['files'][uuid]['group'] for uuid in model['keys']]
    group_mapping = load_groups(index_file, custom_groupings)

    labels = []
    for raw in raw_labels:
        labels.append(group_mapping[raw])

    return sentences, labels
    
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
