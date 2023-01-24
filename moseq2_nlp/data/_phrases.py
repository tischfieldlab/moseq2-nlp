import numpy as np
from moseq2_nlp.util import get_unique_list_elements
from gensim.models.phrases import original_scorer
from brown_clustering import BigramCorpus, BrownClustering
from tqdm import tqdm

class BrownClusterer(object):
    """Object consolidating methjods associated with Brown clustering. Clusters elements in a sequence according to neighborhood statistics. 
    """

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
        return corpus

    def make_brown_tree(self, sentences, alpha=0.0, min_count=0):
        """ Progressively clusters data into larger groups, aggregating each step into a binary tree. 

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

        if not hasattr(self,'clustering'):
            raise ValueError('Sentences have not been clustered. Please run `cluster`.')

        res_codes = [code[:resolution - 1] for code in self.clustering.codes().values()]
        res_dict = {}

        for res_code, (word, code) in zip(res_codes, self.clustering.codes().items()):
            if res_code == code[:resolution-1]:
                res_dict[word] = res_code

        return res_dict

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
        new_sentence = [replacement_dict[word] for word in sentence]
        new_sentences.append(new_sentence)
    return new_sentences

def score_phrases(foreground_seqs, background_seqs, foreground_bigrams, min_count):

    '''score_phrases: assigns a score to each bigram in the foreground sequences based on Mikilov et al., 2013

        Positional args:
            foreground_seqs (list): a list of sequences from which significant phrases are extracted
            background_seqs (list): a list opf sequences against which the foreground sequences are compared for discriminating phrases
            foreground_bigrams (list): a list of precomputed bigrams in the foreground sequences
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
    '''

    # Unique elements in foreground seqs
    unique_foreground_els = get_unique_list_elements(foreground_seqs)

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
        # score = (#(ab in fg) - min) * len_vocab / #(a in bg)*#(b in bg)
                score = original_scorer(count_a, count_b, count_ab, len_vocab, min_count,-1)
                scored_bigrams[bigram] = score

    return scored_bigrams

def make_phrases(foreground_seqs, background_seqs, threshes, n, min_count):

    '''make_phrases: makes a dictionary containing disciminating phrases for a given class

        Positional args:
            foreground_seqs (list): a list of sequences from which significant phrases are extracted
            background_seqs (list): a list opf sequences against which the foreground sequences are compared for discriminating phrases
            threshes (list): a list of floating point thresholds which determine which n-gram phrases will be significant for each n
            n (int): number of times to run the agglomeration algorithm. Running n times will potentially yield up to 2n-grams
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
    '''

    # Flatten list of sequences into one long sequence (introduces artifacts?)
    flat_foreground_seqs = [el for seq in foreground_seqs for el in seq]
    flat_background_seqs = [el for seq in background_seqs for el in seq]

    all_phrases = {}

    # Calculate 2*n grams

    count = 0
    num_syl = float(len(flat_foreground_seqs))
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
            ngram_len = len(bigram.split('>'))
            if score > thresh:
                all_phrases[bigram] = score
                while bigram in foreground_bigrams:
                    count += ngram_len
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
    prop = count / num_syl
    return (all_phrases, prop)

def make_phrases_dataset(sentences, labels, save_path, threshes, n, min_count):

    '''make_phrases_dataset: makes a dictionary containing disciminating phrases for each class in a dataset

        Positional args:
            sentences (List of strs): list of sentences representing moseq emissions
            labels (List of strs): list of class labels for each animal
            save_path (str): path for saving pickled dictionary of phrases
            threshes (list): a list of floating point thresholds which determine which n-gram phrases will be significant for each n
            n (int): number of times to run the agglomeration algorithm. Running n times will potentially yield up to 2n-grams
            min_count (int): minimum number of times a phrase has to appear to be considered for significance
            '''
    # Get labels names
    unique_labels = get_unique_list_elements(labels)
    num_classes = len(unique_labels)
    all_group_phrases = {}

    # For each group
    for label in unique_labels:

        # Compare label to other labels (including itself)
        foreground_sents = [seq for s, seq in enumerate(sentences) if labels[s] == label]
        background_sents = sentences
        all_group_phrases[label] = make_phrases(foreground_sents, background_sents, threshes, n, min_count)

    # Save
    with open(save_path, 'wb') as handle:
        print(f'Saving at {handle}')
        pickle.dump(all_group_phrases, handle, protocol=pickle.HIGHEST_PROTOCOL)

def ablate_phrases(sentences, labels, phrase_path, max_syllable=70):
    with open(phrase_path, 'rb') as fn:
        phrases = pickle.load(fn)

    unique_labels = get_unique_list_elements(labels)

    ablated_sentences = []
    ablated_labels    = []

    for label in unique_labels:
        class_sentences = [sentence for (s, sentence) in enumerate(sentences) if labels[s] == label]
        class_labels = [label] * len(class_sentences)
        class_phrases = phrases[label][0]

        ablated_labels += class_labels

        num_ablated = 0
        total_syl   = 0
        # TODO: make sure you go from long to short phrases
        for sentence in class_sentences:
            total_syl += len(sentence)
            for phrase_tuple in class_phrases:
                phrase = phrase_tuple[0]
                prop = phrase_tuple[1]
                phrase_elements = phrase.split('>')
                k = len(phrase_elements)
                for i in range(len(sentence) - k):
                    candidate_phrase = sentence[i:i+k]
                    if candidate_phrase == phrase_elements:
                        num_ablated += k
                        random_phrase = list(np.random.randint(0, max_syllable, k))
                        sentence[i:i+k] = random_phrase
            ablated_sentences.append(sentence)
        print('%.3f perc. of syllables ablated.' % (100 * (float(num_ablated) / total_syl)) )
    return ablated_sentences, ablated_labels
