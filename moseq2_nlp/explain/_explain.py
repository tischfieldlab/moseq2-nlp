from lime.lime_text import LimeTextExplainer
from moseq2_nlp.models import DocumentEmbedding
from moseq2_nlp.data import get_usage_representation, get_transition_representation
from moseq2_nlp.util import mean_merge_dicts
import re
from random import choice
from tqdm import tqdm

class Explainer(object):
    """Class which collects relevant methods from LimeTextExplainer for integration with Moseq-NLP.

    Methods:
        __init__: instantiates explainer object with all potential feature keyword arguments.
        predict: gets probabilities of sentence features with the provided clasifier.
        explain_instance: returns word weightings from the LIME explainer for a single sentence.
        explain_dataset: returns averaged word weights for all samples in a dataset, organized by class.
        format_sentences: transforms list of space-separated sentence strings into expected moseq-nlp format of list of list of strings.
    """

    def __init__(
        self, feature_name, classifier, class_names, bow=True, custom_feature_map=None, embedding_kwargs=None, **feature_map_kwargs
    ):
        """Instantiates explainer object with all potential feature keyword arguments.

        Args:
            feature_name: str, name of feature map which  maps from sentences to features.
            classifier: sklearn classifier object for classifying feature_map features.
            class_names: list of strings, names of unique classes in dataset.
            bow: bool, if false, differentiates between different appearances of syllable in sentence.
            custom_feature_map: callable, used for feature map if feature_name is not one of `usages`. `transitions`, `embeddings`.
            embedding_kwargs: dict, keyword arguments for doc2vec if feature_name is `embeddings`.
            feature_map_kwargs: dict, dictionary of keyword arguments for feature_map.

        See Also:
            moseq2_nlp.models.DocumentEmbedding
        """
        self.clf = classifier
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names, bow=bow)
        self.set_feature_map(feature_name, embedding_kwargs=embedding_kwargs, custom_feature_map=custom_feature_map)
        self.feature_map_kwargs = feature_map_kwargs

    def predict(self, sentences):
        """Gets probabilities of sentence features with the provided clasifier.

        Args:
            sentences: list of space-separated strings representing sentences.

        Returns:
            np array of class probabilities for all sentences.
        """
        sentences = self.reformat_sentences(sentences)
        features = self.feature_map(sentences, **self.feature_map_kwargs)
        return self.clf.predict_proba(features)

    def explain_instance(self, sentence, num_features=10, num_samples=1000):
        """Returns word weightings from the LIME explainer for a single sentence.

        Args:
            sentence: string of space-separated ints representing moseq syllables.
            num_features: int, maximum number of features used in the explanation
            num_samples: number of randomly ablated versions of sentence to use to approximate feature landscape.

        Returns:
            LIME explainer object with weightings per syllable.

        See Also:
            LimeTextExplainer
        """
        return self.explainer.explain_instance(sentence, self.predict, num_features=num_features, num_samples=num_samples)

    def explain_dataset(self, sentences, labels, num_features=10, num_samples=1000):
        """Returns word weightings from the LIME explainer for a full data set, averaged per class.

        Args:
            sentences: list of string of space-separated ints representing moseq syllables.
            num_features: int, maximum number of features used in the explanation
            num_samples: number of randomly ablated versions of sentence to use to approximate feature landscape.

        Returns:
            list of LIME explainer objects per class.

        See Also:
            LimeTextExplainer
        """
        class_explanations = {}
        for unique_label in tqdm(self.class_names):
            class_sentences = [sentence for i, sentence in enumerate(sentences) if labels[i] == unique_label]
            instance_expls = [
                {
                    k: v
                    for (k, v) in self.explainer.explain_instance(
                        sentence, self.predict, num_features=num_features, num_samples=num_samples
                    ).as_list()
                }
                for sentence in class_sentences
            ]
            instance_expls_dict = mean_merge_dicts(instance_expls)
            class_explanations[unique_label] = instance_expls_dict

        return class_explanations

    def reformat_sentences(self, sentences):
        """Transforms list of space-separated sentence strings into expected moseq-nlp format of list of list of strings.

        Args:
            sentences: list of string of space-separated ints representing moseq syllables to be reformatted.

        Returns:
            sentences: list of list of syllables (strings)
        """
        list_sentences = [sentence.split() for sentence in sentences]
        if "max_syllable" in self.feature_map_kwargs.keys():
            max_syllable = self.feature_map_kwargs["max_syllable"]
            list_sentences = [list(filter(lambda s: int(s) <= max_syllable, sentence)) for sentence in list_sentences]

        if "bad_syllables" in self.feature_map_kwargs.keys():
            bad_syllables = self.feature_map_kwargs["bad_syllables"]
            list_sentences = [list(filter(lambda s: int(s) not in bad_syllables, sentence)) for sentence in list_sentences]

        n_vocabs = [len(list(set(sentence))) for sentence in list_sentences]
        good_sentences = [sentence for i, sentence in enumerate(list_sentences) if n_vocabs[i] > 3]

        formatted_sentences = []
        for n_vocab, sentence in zip(n_vocabs, list_sentences):
            n_vocab = len(list(set(sentence)))
            if n_vocab > 3:
                formatted_sentences.append(sentence)
            else:
                formatted_sentences.append(choice(good_sentences))
        return formatted_sentences

    def set_feature_map(self, feature_name, embedding_kwargs=None, custom_feature_map=None):
        """Sets the feature map used by the classifier.

        Args:
            feature_name: str, the name of the feature map
            embedding_kwargs: dict, if feature_name = `embeddings`, this dict contains the kwargs for the doc2vec embeddings
            custom_feature_map: callable, if feature_map is not one of `usages`, `transitions` or `embeddings`, then this is used as the feature map
        """
        if feature_name == "usages":
            self.feature_map = get_usage_representation
        elif feature_name == "transitions":
            self.feature_map = get_transition_representation
        elif feature_name == "embeddings":
            dm = embedding_kwargs["dm"]
            embedding_dim = embedding_kwargs["embedding_dim"]
            embedding_window = embedding_kwargs["embedding_window"]
            embedding_epochs = embedding_kwargs["embedding_epochs"]
            min_count = embedding_kwargs["min_count"]
            negative = embedding_kwargs["negative"]
            seed = embedding_kwargs["seed"]
            model_path = embedding_kwargs["model_path"]

            # Instantiate and load model
            model = DocumentEmbedding(
                dm=dm,
                embedding_dim=embedding_dim,
                embedding_window=embedding_window,
                embedding_epochs=embedding_epochs,
                min_count=min_count,
                negative=negative,
                seed=seed,
            )

            model.load(model_path)

            # Define embedding function
            self.feature_map = model.predict
        else:
            self.feature_map = custom_feature_map
