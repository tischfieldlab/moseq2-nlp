from configparser import ConfigParser
import argparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from  models import DocumentEmbedding
from utils import load_data
from tqdm import tqdm
import json
import numpy as np
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='DEFAULT')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

# Load experimental parameters
config = ConfigParser(allow_no_value=True)
config.optionxform = str
config.read('config.cfg')
config_dict = {}
for (key, val) in config.items(args.name):
    config_dict[key] = val

data_dir           = config_dict['data_dir']
save_dir           = config_dict['save_dir']
experiment         = config_dict['experiment']
timepoint          = int(config_dict['timepoint'])
representation     = config_dict['representation']
emissions          = config.getboolean(args.name, 'emissions')
custom_labels      = config_dict['custom_labels'].split(',')
custom_label_names = config_dict['custom_label_names'].split(',')
max_syllable       = int(config_dict['max_syllable'])
num_transitions    = int(config_dict['num_transitions'])
min_count          = int(config_dict['min_count'])
dm                 = int(config_dict['dm'])
embedding_dim      = int(config_dict['embedding_dim'])
embedding_window   = int(config_dict['embedding_window'])
embedding_epochs   = int(config_dict['embedding_epochs'])
bad_syllables      = config_dict['bad_syllables'].split(',')
bad_syllables      = [int(bs) for bs in bad_syllables]
scoring            = config_dict['scoring']
K                  = int(config_dict['K'])
penalty            = config_dict['penalty']
num_C              = int(config_dict['num_C'])

exp_dir = os.path.join(save_dir, args.name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
   
print('Getting data') 
labels, usages, transitions, sentences, bigram_sentences = load_data(data_dir,
                                                       experiment, 
                                                       emissions=emissions,
                                                       custom_labels=custom_labels,
                                                       custom_label_names=custom_label_names,
                                                       max_syllable=max_syllable,
                                                       num_transitions=num_transitions,
                                                       bad_syllables=bad_syllables,
                                                       timepoint=timepoint)

print('Getting features')
num_animals = len(labels) 
if representation == 'embeddings':
    model  = DocumentEmbedding(dm=dm, embedding_dim=embedding_dim, embedding_window=embedding_window, embedding_epochs=embedding_epochs, min_count=min_count)
    rep = np.array(model.fit_predict(sentences))
elif representation == 'usages':
    rep = usages
elif representation == 'transitions':
    rep = transitions
else:
    raise ValueError('Representation type not recognized. Valid values are "usages", "transitions" and "embeddings".')

print('Training classifier')
Cs = np.logspace(-5,5,num_C)
kf = KFold(n_splits=int(num_animals / float(K)))
# Load and train classifier
if penalty is not 'none':
    clf = LogisticRegressionCV(Cs=Cs, cv=kf, scoring=scoring,random_state=args.seed, dual=False, solver='lbfgs', penalty=penalty,class_weight='balanced',multi_class='auto', tol=1e-6, max_iter=2000).fit(rep,labels)
else:
    clf = LogisticRegressionCV(cv=kf, scoring=scoring,random_state=args.seed, dual=False, solver='lbfgs', class_weight='balanced', multi_class='auto', tol=1e-6, max_iter=2000).fit(rep,labels)
scores = np.array([sc for sc in clf.scores_.values()]) # nm_classes x num_folds x num_C
best_score = np.max(scores.mean((0,1)))
best_C     = Cs[np.argmax(scores.mean((0,1)))]

print('Best {}: {}'.format(scoring, best_score))
print('Best C: {}'.format(best_C))

fn = os.path.join(exp_dir, 'exp_params.txt')
with open(fn, 'w') as file:
    file.write(json.dumps(config_dict))
np.save(os.path.join(exp_dir, '{}.npy'.format(scoring)), best_score)
np.save(os.path.join(exp_dir, 'best_C.npy'), best_C)
