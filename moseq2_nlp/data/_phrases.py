import numpy as np
from moseq2_nlp.util import get_unique_list_elements, ensure_dir
from gensim.models.phrases import Phrases
from brown_clustering import BigramCorpus, BrownClustering
from tqdm import tqdm
import pickle
import os

class BrownClusterer(object):
    """Object consolidating methjods associated with Brown clustering. Clusters elements in a sequence according to neighborhood statistics."""

    def make_corpus(self, sentences, alpha=0.0, min_count=0):
        """Converts sentences to a bigram corpus object.

        Args:
            sentences: a list of list of strings. Each sublist contains all of the syllables for an animal.
            alpha: float controling degree of Laplacian smoothing.
            min_count: int indicating the minimum number of instances a syllable must have to be included in the corpus

        Returns:
            corpus: BigramCorpus object
        """
        corpus = BigramCorpus(sentences, alpha=alpha, min_count=min_count)

        self.corpus = corpus
        self.n_vocab = len(self.corpus.vocabulary)
        return corpus

    def make_brown_tree(self, sentences, alpha=0.0, min_count=0):
        """Progressively clusters data into larger groups, aggregating each step into a binary tree.

        Args:
            sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
            alpha: float controling degree of Laplacian smoothing.
            min_count: int indicating the minimum number of instances a syllable must have to be included in the corpus
        """
        corpus = self.make_corpus(sentences, alpha=alpha, min_count=min_count)

        num_vocab = len(corpus.vocabulary)

        clustering = BrownClustering(corpus, m=num_vocab)

        self.clustering = clustering

        self.clustering.train()

    def get_clusters_by_resolution(self, resolution):
        """Returns a clustering of sentence data at a given depth of the Brown tree. Higher resolution means more clusters.

        Args:
            resolution: level at which to read a clustering. Higher means more clusters.

        Returns:
            res_dict: dictionary which maps from a syllable name to its cluster id at the given resolution
        """
        if not hasattr(self, "clustering"):
            raise ValueError("Sentences have not been clustered. Please run `cluster`.")

        res_codes = [code[: resolution - 1] for code in self.clustering.codes().values()]
        res_dict = {}

        for res_code, (word, code) in zip(res_codes, self.clustering.codes().items()):
            if res_code == code[: resolution - 1]:
                res_dict[word] = res_code

        return res_dict

def save_brown_datasets(sentences, labels, save_dir, alpha=.5, min_count=0):
    """Finds synonyms in a dataset of sentences and then saves clustered versions at different resolutions.

    Args:
        sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
        labels: list, labels to save with each clustered dataset
        save_dir: str, where to save each of the clustered datasets 
        alpha: float controling degree of Laplacian smoothing.
        min_count: int indicating the minimum number of instances a syllable must have to be included in the corpus.
    """
    # Instantiate BC
    bc = BrownClusterer()

    # Make tree
    print('Finding clusters.')
    bc.make_brown_tree(sentences, alpha=alpha, min_count=min_count)
   
    current_clusters = -1
    print('Saving Brown clustered data.')
    for resolution in tqdm(np.arange(1, bc.n_vocab)):
        res_clusters = bc.get_clusters_by_resolution(resolution)
        num_clusters = len(get_unique_list_elements(res_clusters.values()))

        if num_clusters == current_clusters:
            print(f'Saved {resolution} clustered datasets.')
            break
        else:
            current_clusters = num_clusters
            new_sentences = replace_words(sentences, res_clusters)

            # Make dir
            res_dir = os.path.join(save_dir, f'data_resolution_{resolution}')
            ensure_dir(res_dir)
                
            # Save
            names = ['sentences', 'labels', 'cluster_map'] 
            res_clusters = [(k,v) for (k,v) in res_clusters.items()]
            for obj, nm in zip([new_sentences, labels, res_clusters], names):
                res_path = os.path.join(res_dir, f'{nm}.pkl')
                with open(res_path, "wb") as handle:
                    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def replace_words(sentences, replacement_dict):
    """Replaces the symbols of a sentence according to the provided mapping. Can be used in conjunction with a phrasing algorithm to consolidate words into composite symbols.

    Args:
        sentences: a list of list of strings. Each sublist contains all of the syllables for an animal. The full list contains all animals.
        replacement_dict: a dictionary which maps from the sentence syllable to new symbols.

    Returns:
        new_sentences: sentences with replaced symbols
    """
    new_sentences = []

    for sentence in sentences:
        new_sentence = []
        for word in sentence:
            if word in replacement_dict.keys():
                new_sentence.append(replacement_dict[word])
            else:
                new_sentence.append(word)
        new_sentences.append(new_sentence)
    return new_sentences

def find_phrases(sentences, min_count=1, threshold=1.0, scoring='default'):
    """Finds and returns a phrase model based on statistics from `sentences`.

    Args:
        sentences: list of list of strings, sentences in which to detect phrases.
        min_count: int, minimum number of times a phrase has to appear to be included in phrase list
        threshold: float, threshold for inclusion in phrases. Interpretation depends on scorer
        scoring: str, one of two types of scoring methods, `default` or `npmi`

    Returns:
        Phrases: a gensim phrase model object containing information about phrases.

    See Also:
        gensim.models.phrases
    """
    return Phrases(sentences, min_count=min_count, threshold=threshold, scoring=scoring)

def save_phrase_datasets(sentences, thresholds, save_dir, iterations=1, min_count=1, scoring='default'):
    """Iteratively groups words into phrases and saves each iteration as a dataset.

    Args:
        sentences: list of list of strings, sentences in which to detect phrases.
        thresholds: list of floats, thresholds for inclusion in phrases per iteration. Interpretation depends on scorer
        save_dir: str, where to save all of the phrased datasets. 
        iterations: int, number of passes of the phraser.
        min_count: int, minimum number of times a phrase has to appear to be included in phrase list
        scoring: str, one of two types of scoring methods, `default` or `npmi`

    """
    print('Finding phrases.')
    for i in tqdm(range(iterations)):
        phrase_model = find_phrases(sentences, min_count=min_count, threshold=thresholds[i], scoring=scoring)
        sentences = [phrase_model[sentence] for sentence in sentences]

        iter_dir = os.path.join(save_dir, f'phrase_iterations_{i + 1}')
        ensure_dir(iter_dir)

        phrase_path = os.path.join(iter_dir, 'sentences.pkl')
        with open(phrase_path, "wb") as handle:
            pickle.dump(sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)
