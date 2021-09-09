import time
times = {'Preamble' : 0.0, 'Data' : 0.0, 'Features' :0.0, 'Classifier' : 0.0}
start = time.time()
import configargparse
from distutils.util import strtobool
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from  models import DocumentEmbedding
from moseq2_nlp.utils import load_data
from tqdm import tqdm
import json
import numpy as np
import time
import os

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
Cs = np.logspace(-5,5,args.num_C)
kf = KFold(n_splits=int(num_animals / float(args.K)))
# Load and train classifier
if args.penalty != 'none':
    clf = LogisticRegressionCV(Cs=Cs, cv=kf, scoring=args.scoring,random_state=args.seed, dual=False, solver='lbfgs', penalty=args.penalty,class_weight='balanced',multi_class='auto', tol=1e-6, max_iter=2000).fit(rep,labels)
else:
    clf = LogisticRegressionCV(cv=kf, scoring=args.scoring,random_state=args.seed, dual=False, solver='lbfgs', class_weight='balanced', multi_class='auto', tol=1e-6, max_iter=2000).fit(rep,labels)
scores = np.array([sc for sc in clf.scores_.values()]) # nm_classes x num_folds x num_C
best_score = np.max(scores.mean((0,1)))
best_C     = Cs[np.argmax(scores.mean((0,1)))]
times['Classifier'] = time.time() - start

print('Best {}: {}'.format(args.scoring, best_score))
print('Best C: {}'.format(best_C))

save_dict = args.__dict__
save_dict['compute_times']=times
fn = os.path.join(exp_dir, 'exp_params.txt')
with open(fn, 'w') as file:
    file.write(json.dumps(save_dict))
np.save(os.path.join(exp_dir, '{}.npy'.format(args.scoring)), best_score)
np.save(os.path.join(exp_dir, 'best_C.npy'), best_C)
