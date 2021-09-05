import os
import numpy as np
from gensim.models import Phrases
import sys
moseq_path = '/media/data_cifs/matt/moseq2-viz'
sys.path.append(moseq_path)
from tqdm import tqdm
import pdb
import moseq2_viz
from moseq2_viz.util import parse_index
from moseq2_viz.model.util import (get_transition_matrix,
                                   parse_model_results,
                                   results_to_dataframe,
                                   relabel_by_usage, get_syllable_statistics)
def set_data(super_dir, experiment, **kwargs):
    max_syllable = kwargs['max_syllable']
    num_transitions=kwargs['num_transitions']

    if experiment == '2021-02-19_Meloxicam':
        model_file = os.path.join(super_dir, experiment, 'rST_model_1000.p')
        index_file = os.path.join(super_dir, experiment, 'moseq2-index.role.yaml')
        groups = ['baseline', '4hrs carrageenan', '24hrs saline', '24hrs meloxicam', 'baseline meloxicam']
    elif experiment == 'WWC1_W88C':
        model_file = os.path.join(super_dir, experiment, 'robust_septrans_model_500.p')
        index_file = os.path.join(super_dir, experiment, 'moseq2-index.sex-genotype.yaml')
        groups = ['M_WC/WC', 'F_WC/WC', 'M_+/WC', 'F_+/WC', 'M_+/+', 'F_+/+']
        ## I AM LEAVIG OUT M_ukn!
            
    elif experiment == '2019-07-12_Junbing_Bicuculline_Celsr3-PvCre':
        model_file = os.path.join(super_dir, experiment, 'first_model.p')
        index_file = os.path.join(super_dir, experiment, 'moseq2-index-cohort-sex-role.yaml')
        groups = ['default', 'Bicuculline_Control_UKN', 'Bicuculline_Experimental_UKN',
                  'ctrl_wt_control_Male', 'ctrl_wt_control_Female',
                  'Celsr3_PvCre_Control_Male', 'Celsr3_PvCre_Control_Female',
                  'Celsr3_PvCre_Experimental_Male', 'Celsr3_PvCre_Experimental_Female',
                  'Celsr3_PvCre_Het_Female','Celsr3_PvCre_Het_Male']
        
        BMS_Left = []
        BMS_Right = []
        with open(os.path.join(super_dir, experiment, "session_manifest.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for r, row in enumerate(rd):
                if r == 0 or (len(row) < 17): continue
                BMS_Left.append(int(row[-3]))
                BMS_Right.append(int(row[-2]))
            
    elif experiment == '2020-11-10_Celsr3-R774H':
        model_file = os.path.join(super_dir, experiment, 'robust_septrans_model_1000.p')
        index_file = os.path.join(super_dir, experiment, 'gender-genotype-index.yaml')
        groups = ['F_+/+', 'F_RH/RH', 'F_+/RH', 'M_+/+', 'M_RH/RH', 'M_+/RH']
            
    elif experiment == 'SstCre_Celsr3CKO':
        model_file = os.path.join(super_dir, experiment, 'robust_septrans_model_20min_1000.p')
        index_file = os.path.join(super_dir, experiment, 'moseq2-index.sex-genotype.20min.yaml')
        groups = ['F_+/+;Celsr3^f/f', 'F_Sst-Cre/+;Celsr3^f/f', 'M_Sst-Cre/+;Celsr3^f/f', 'M_+/+;Celsr3^f/f']
            
    elif experiment == '2019SCI':
        model_file = os.path.join(super_dir, experiment, 'second_model.p')
        index_file = os.path.join(super_dir, experiment, 'moseq2-index.timepoint.yaml')
        groups = ['before SCI'] + ['{} week after SCI'.format(w) for w in range(2,8)]
        BMS_Left = []
        BMS_Right = []
        with open(os.path.join(super_dir, experiment, "cohort_manifest.tsv")) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for r, row in enumerate(rd):
                if r == 0 or (len(row) < 17): continue
                BMS_Left.append(int(row[-3]))
                BMS_Right.append(int(row[-2]))
                
    elif experiment == '2021-06-04_P4_C2_SCI_timecourse':
        timepoint = kwargs['timepoint']
        model_file = os.path.join(super_dir, experiment, 'P{}-sex_rST_model_1000.p'.format(timepoint)) 
        index_file = os.path.join(super_dir, experiment, 'moseq2-index.P{}.grouped-sex.yaml'.format(timepoint))
        
        groups = ['P4_C2_SCI_P{}_M', 'P4_C2_SCI_P{}_F','Control_P{}_M','Control_P{}_F']
        custom_labels = [0,0,1,1]
        custom_label_names = ['Injury', 'Control']
        for g in range(len(groups)):
            groups[g] = groups[g].format(timepoint)
        if timepoint == '10':
            groups.remove( 'P4_C2_SCI_P10_F')
        elif timepoint == '60' or timepoint=='90':
            groups.remove('Control_P{}_F'.format(timepoint))
    return model_file, index_file, groups

def load_data(super_dir, experiment, **kwargs):
    emissions = kwargs['emissions']
    max_syllable = kwargs['max_syllable']
    num_transitions=kwargs['num_transitions']
    bad_syllables=kwargs['bad_syllables']
    custom_labels = kwargs['custom_labels']
    custom_label_names = kwargs['custom_label_names']

    model_file, index_file, groups = set_data(super_dir, experiment, **kwargs)
    _, sorted_index = parse_index(index_file)

    ms_model = parse_model_results(model_file, sort_labels_by_usage=True, count='usage')
    labels = ms_model['labels']
    label_group = [sorted_index['files'][uuid]['group'] for uuid in ms_model['keys']]
    
    use_BMS_custom_labels = False
    bms_custom_labels = [0,0,0,1,1,1,2,2,2]
    tm_vals = []
    truncated_tm_vals = []
    group_vals = []
    group_labels = []
    usage_vals = []
    frames_vals = []
    sentences = []
    bigram_sentences = []
    sentence_strings = []
    sentence_groups = {group : [] for group in groups}
    for i, (l, g) in tqdm(enumerate(zip(labels, label_group))):
        
        if g not in groups and g != 'M_ukn':
            raise ValueError('Group name in data not recognized. Check the group names you specified!')
        elif g == 'M_ukn':
            continue
        group_vals.append(g)
    
        # Label data using default or custom labels
        group_labels.append(custom_labels[groups.index(g)])
    
        # Get transitions
        tm = get_transition_matrix([l], combine=True, max_syllable=max_syllable - 1)
        tm_vals.append(tm.ravel())
    
        # Get usages
        u, _ = get_syllable_statistics(l, count='usage')
        u_vals = list(u.values())[:max_syllable]
        total_u = np.sum(u_vals)
        usage_vals.append(np.array(u_vals) / total_u)
    
        # Get frame values
        f, _ = get_syllable_statistics(l, count='usage')
        total_f = np.sum(list(f.values()))
        frames_vals.append(np.array(list(f.values())) / total_f)
    
        # Get emissions
        l = list(filter(lambda a: a not in bad_syllables, l))
        np_l = np.array(l)
        if emissions:
            cp_inds = np.concatenate((np.where(np.diff(np_l) != 0 )[0],np.array([len(l) - 1])))
            syllables = np_l[cp_inds]
        else:
            syllables = np_l
        sentence = [str(syl) for syl in syllables]
        sentences.append(sentence)
        sentence_strings.append(' '.join(sentence))
        sentence_groups[g].append(sentence)
    
        bigram_model = Phrases(sentence, min_count=1, threshold=1, scoring='default')
        bgs = bigram_model[sentence]
    #     print(len([phr for phr in bgs if '_' in phr]))
        bigram_sentences.append(bgs)
            
    # Post-processing including truncation of transitions
    # Truncated transitions
    tm_vals = np.array(tm_vals)
    top_transitions = np.argsort(tm_vals.mean(0))[-num_transitions:]
    truncated_tm_vals = tm_vals[:,top_transitions]
    
    # Make numpy
    usage_vals = np.array(usage_vals)
    frames_vals = np.array(frames_vals)
    num_animals = len(sentences)
    
    np_g = np.array(group_labels)
    group_sizes = [sum(np_g == g) for g in np.unique(np_g)]
    lb_ind = np.argsort(np_g)
    return group_labels, usage_vals, truncated_tm_vals, sentences, bigram_sentences
