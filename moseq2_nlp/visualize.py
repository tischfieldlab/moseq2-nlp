import numpy as np
import pdb
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
from gensim.models.doc2vec import Doc2Vec
from models import DocumentEmbedding

num_syllables = 70
fn = '/media/data_cifs/matt/abraira_results/tmp/doc2vec'
model = DocumentEmbedding(dm=2)
model.load(fn)
words = model.model0.wv.key_to_index.keys()
rep = np.array([model.predict_word([word]) for word in words])
pdb.set_trace()
pca = PCA(n_components=2)
result = pca.fit_transform(rep)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
plt.savefig('/media/data_cifs/matt/tmp.png')
plt.close()
