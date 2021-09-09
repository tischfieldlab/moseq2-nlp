from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

class DocumentEmbedding(object):
    def __init__(self, dm=0, embedding_dim=256, embedding_window=5, embedding_epochs=50, min_count=5):
        self.dm = dm
        self.embedding_dim = embedding_dim
        self.embedding_epochs = embedding_epochs
        self.embedding_window = embedding_window
        self.min_count = min_count

    def fit(self, sentences):
        documents = [TaggedDocument(sent, [i]) for i, sent in enumerate(sentences)]
        if self.dm < 2 and self.dm >=0:
            self.model = Doc2Vec(documents, dm=self.dm, epochs=self.embedding_epochs, vector_size=self.embedding_dim, window=self.embedding_window, min_count=self.min_count, workers=1)
        elif self.dm == 2:
            self.model0 = Doc2Vec(documents, dm=0, epochs=self.embedding_epochs, vector_size=self.embedding_dim, window=self.embedding_window, min_count=self.min_count, workers=1, dbow_words=0)
            self.model1 = Doc2Vec(documents, dm=1, epochs=self.embedding_epochs, vector_size=self.embedding_dim, window=self.embedding_window, min_count=self.min_count, workers=1)
        else:
            raise ValueError('Distributed memory value not valid. Accepted values are dm=0,1,2.')

    def predict(self, sentences):
        if self.dm < 2 and self.dm >=0:
            return np.array([self.model.infer_vector(sent) for sent in sentences])
        elif self.dm == 2:
            E0 = [self.model0.infer_vector(sent) for sent in sentences]
            E1 = [self.model1.infer_vector(sent) for sent in sentences]
            return np.array([.5 * (em0 + em1) for (em0, em1) in zip(E0, E1)])
        else:
            raise ValueError('Distributed memory value not valid. Accepted values are dm=0,1,2.')

    def predict_word(self, word):
        e0 = self.model0[word]
        e1 = self.model1[word]
        return .5*(e0 + e1)

    def fit_predict(self, sentences):
        self.fit(sentences)
        return self.predict(sentences)

    def save(self, name):
        if self.dm < 2:
            self.model.save(f'{name}.model')
        else:
            self.model0.save(f'{name}.0.model')
            self.model1.save(f'{name}.1.model')

    def load(self, name):
        if self.dm < 2:
            self.model = Doc2Vec(f'{name}.model')
        else:
            self.model0 = Doc2Vec.load(f'{name}.0.model')
            self.model1 = Doc2Vec.load(f'{name}.1.model')
