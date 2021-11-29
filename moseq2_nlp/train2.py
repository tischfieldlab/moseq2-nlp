import time
times = {'Preamble' : 0.0, 'Data' : 0.0, 'Features' :0.0, 'Classifier' : 0.0}
start = time.time()
import configargparse
from distutils.util import strtobool
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from  models import DocumentEmbedding
from moseq2_nlp.utils import load_data
from utils import filter_corpora
from tqdm import tqdm
import json
import numpy as np
import time
import os
import pdb

parser = configargparse.ArgParser(default_config_files=['./config.cfg'])

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--name', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--index_path', type=str)
parser.add_argument('--representation', type=str)
parser.add_argument('--emissions',type= lambda x:bool(strtobool(x)))
parser.add_argument('--custom_groupings',action='append', nargs='?')
parser.add_argument('--num_syllables',type=int)
parser.add_argument('--num_transitions',type=int)
parser.add_argument('--min_count',type=int)
parser.add_argument('--dm',type=int)
parser.add_argument('--embedding_dim',type=int)
parser.add_argument('--embedding_window',type=int)
parser.add_argument('--embedding_epochs',type=int)
parser.add_argument('--bad_syllables',action='append')
parser.add_argument('--scoring',type=str)
parser.add_argument('--K',type=int)
parser.add_argument('--penalty',type=str)
parser.add_argument('--num_C',type=int)
parser.add_argument('--preprocessing',type=str)
args = parser.parse_args()
if args.custom_groupings is not None:
    custom_groupings = [s.split(',') for s in args.custom_groupings]
else:
    custom_groupings = []
if args.bad_syllables is not None:
    bad_syllables = [int(bs) for bs in args.bad_syllables]
else:
    bad_syllables = []
exp_dir = os.path.join(args.save_dir,args.name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
times['Preamble'] = time.time() - start

start = time.time()
print('Getting data') 
labels, usages, transitions, sentences, bigram_sentences = load_data(args.model_path,
                                                       args.index_path,
                                                       emissions=args.emissions,
                                                       custom_groupings=custom_groupings,
                                                       num_syllables=args.num_syllables,
                                                       num_transitions=args.num_transitions,
                                                       bad_syllables=bad_syllables)

if args.preprocessing == 'targeted_ablation':
    sentences = filter_corpora(sentences, labels, 4, 5, args.num_syllables,
                               [.001, .0005, .0005], bad_syllables, replacement='random')
elif args.preprocessing == 'random_ablation':
    # Good: .05, .0001 .0001
    sentences = filter_corpora(sentences, labels,vocab_size=args.num_syllables, bad_syllables=bad_syllables, filtering='random', replacement='random')

times['Data'] = time.time() - start

start = time.time()
print('Getting features')
num_animals = len(labels) 
if args.representation == 'embeddings':
    model  = DocumentEmbedding(dm=args.dm, embedding_dim=args.embedding_dim, embedding_window=args.embedding_window, embedding_epochs=args.embedding_epochs, min_count=args.min_count)
    rep = np.array(model.fit_predict(sentences))
    #model.save(os.path.join(exp_dir, 'doc2vec'))
elif args.representation == 'usages':
    rep = usages
elif args.representation == 'transitions':
    rep = transitions
else:
    raise ValueError('Representation type not recognized. Valid values are "usages", "transitions" and "embeddings".')
times['Features'] = time.time() - start

start = time.time()
print('Training classifier')

all_pred = []
all_test = []
C=10000
seed=0
for i in range(int(num_animals / float(args.K))):
    shifted_rep = np.roll(rep, i*args.K, axis=0)
    shifted_labels = np.roll(labels, i*args.K, axis=0)

    # Split into train and test
    train_X, train_y = shifted_rep[args.K:,:], shifted_labels[args.K:]
    test_X, test_y = shifted_rep[:args.K,:], shifted_labels[:args.K]

    clf = LogisticRegression(random_state=seed,dual=False,solver='lbfgs', penalty=args.penalty,class_weight='balanced', multi_class='auto',C=C, tol=1e-6, max_iter=2000).fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    all_pred.append(y_pred)
    all_test.append(test_y)

np.save('/media/data_cifs/matt/all_pred_U.npy',all_pred)
np.save('/media/data_cifs/matt/all_test_U.npy',all_test)
print('done')
