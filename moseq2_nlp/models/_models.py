from typing import List, Literal

import numpy as np

import gensim.models.doc2vec

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class DocumentEmbedding(object):
    def __init__(
        self,
        dm: Literal[0, 1, 2] = 0,
        embedding_dim: int = 256,
        embedding_window: int = 5,
        embedding_epochs: int = 50,
        min_count: int = 2,
        negative: int = 0,
        seed: int = 0,
        multithreading: bool = False,
    ):
        """Create a document emedding with some parameters

        Parameters:
            dm (int): Defines the training algorithm. If dm=1, 'distributed memory' (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
            embedding_dim (int): Dimensionality of the feature vectors.
            embedding_window (int): The maximum distance between the current and predicted word within a sentence.
            embedding_epochs (int): Number of iterations (epochs) over the corpus.
            min_count (int): Ignore all words with total frequency lower than this.
        """
        self.dm = dm
        self.embedding_dim = embedding_dim
        self.embedding_epochs = embedding_epochs
        self.embedding_window = embedding_window
        self.min_count = min_count
        self.seed = seed

    def fit(self, sentences: List[List[str]]) -> None:
        """Fit a model to some data.

        Parameters:
            sentences (List[List[str]]): data to fit the model to
        """
        documents = [TaggedDocument(sent, [i]) for i, sent in enumerate(sentences)]
        if self.dm < 2 and self.dm >= 0:
            self.model = Doc2Vec(
                documents,
                dm=self.dm,
                epochs=self.embedding_epochs,
                vector_size=self.embedding_dim,
                window=self.embedding_window,
                min_count=self.min_count,
                workers=1,
                seed=self.seed,
            )
        elif self.dm == 2:
            self.model0 = Doc2Vec(
                documents,
                dm=0,
                epochs=self.embedding_epochs,
                vector_size=self.embedding_dim,
                window=self.embedding_window,
                min_count=self.min_count,
                workers=1,
                dbow_words=0,
                seed=self.seed,
            )
            self.model1 = Doc2Vec(
                documents,
                dm=1,
                epochs=self.embedding_epochs,
                vector_size=self.embedding_dim,
                window=self.embedding_window,
                min_count=self.min_count,
                workers=1,
                seed=self.seed,
            )
        else:
            raise ValueError("Distributed memory value not valid. Accepted values are dm=0,1,2.")

    def predict(self, sentences: List[List[str]]) -> np.ndarray:
        """Predict.

        Parameters:
            sentences (List[List[str]]): data to fit the model to

        Returns:
            (np.ndarray):
        """
        if self.dm < 2 and self.dm >= 0:
            return np.array([self.model.infer_vector(sent) for sent in sentences])
        elif self.dm == 2:
            E0 = [self.model0.infer_vector(sent) for sent in sentences]
            E1 = [self.model1.infer_vector(sent) for sent in sentences]
            return np.array([0.5 * (em0 + em1) for (em0, em1) in zip(E0, E1)])
        else:
            raise ValueError("Distributed memory value not valid. Accepted values are dm=0,1,2.")

    def predict_word(self, word: str) -> np.ndarray:
        """Predict a single word.

        Parameters:
            word (str):

        Returns:
            (np.ndarray):
        """
        e0 = self.model0[word]
        e1 = self.model1[word]
        return 0.5 * (e0 + e1)

    def fit_predict(self, sentences) -> np.ndarray:
        """Fit the model and then predict results.

        Equivelent to:
        ```
        model.fit(sentences)
        model.predict(sentences)
        ```

        Parameters:
            sentences: data to fit and predict
        """
        self.fit(sentences)
        return self.predict(sentences)

    def save(self, name: str) -> None:
        """Save this model to disk.

        `name` should be directory path plus the base name of the file.
        Appropriate file extensions will be added to the supplied name.

        Example:
        ```
        model.save('/path/to/dest/my-model')
        ```
        will result in the file (when `dm < 2`):
        `/path/to/dest/my-model.model`

        Or the files (when `dm == 2`):
        `/path/to/dest/my-model.0.model` and `/path/to/dest/my-model.1.model`

        Parameters:
            name (str): path (incliding basename) to where the model should be saved
        """
        if self.dm < 2:
            self.model.save(f"{name}.model")
        else:
            self.model0.save(f"{name}.0.model")
            self.model1.save(f"{name}.1.model")

    def load(self, name: str):
        """Load a previously saved model from disk.

        Path should be directory path plus the base name of the file.
        Appropriate file extensions will be added to the supplied name.

        Example:
        ```
        model.load('/path/to/dest/my-model')
        ```

        Parameters:
            name (str): path (incliding basename) to where the model should be saved
        """
        if self.dm < 2:
            self.model = Doc2Vec.load(f"{name}.model")
        else:
            self.model0 = Doc2Vec.load(f"{name}.0.model")
            self.model1 = Doc2Vec.load(f"{name}.1.model")
