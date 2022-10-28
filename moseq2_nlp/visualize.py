import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle
from collections import ChainMap
import matplotlib.pyplot as plt
#from moseq2_nlp.data import get_raw_data
from matplotlib.lines import Line2D
import matplotlib
#matplotlib.rc('font', 'Arial')
from tqdm import tqdm
import os
import pdb

class SimpleGroupedColorFunc(object):
    '''SimpleGroupedColorfunc: class used for making a color dictionary for wordcloud. Taken from wordcloud docs. '''
    def __init__(self,color_to_words,default_color):
        self.word_to_color = {word: color for (color, words) in color_to_words.items() for word in words}
        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)

def make_wordcloud(phrases_path, save_dir, max_plot=15):

    '''make_wordcloud: takes a dictionary of phrases and turns them into a wordcloud
    
       Positional args:
           phrases_path (str): the location of the phrase dictionary
           save_dir (str): location of the directory where wordcloud will be saved
        
       Keyword args:
           max_plot (int, default=15): maximum number of phases to plot per class'''

    # Load phrases
    with open(phrases_path, 'rb') as handle:
        group_dict = pickle.load(handle)

    all_colors = ['red', 'blue', 'green', 'orange', 'purple', 'grey', 'black', 'pink', 'forestgreen', 'cornflower', 'magenta', 'cyan']
   
    legend_elements = []

    full_phrase_dict = {}
    color_dict = {}

    my_flag = False

    for key in group_dict.keys():
        if 'WC' in key:
            my_flag = True

    # For each group dictionary
    for g, (group, (phrase_dict, _)) in enumerate(group_dict.items()):

        if '_F' in group:
            continue

        scores    = [score for score in phrase_dict.values()]
        phrases   = [phrase for phrase in phrase_dict.keys()]
        num_plot = min(len(phrases), max_plot)

        # Get sorted score indices from highest to lowest with associated phrases
        sorted_scores = np.argsort(scores)[::-1][:num_plot]
        sorted_keys = [phrases[s] for s in sorted_scores]

        # Disambiguate keys in full phrase dictionary
        for k, key in enumerate(sorted_keys):
            base_key = key.split(' ')[0]
            full_phrase_keys = [k1 for k1 in full_phrase_dict.keys()]
            full_base_keys = [ky2.split(' ')[0] for ky2 in full_phrase_keys]
            if base_key in full_base_keys:
                num_stars = 0
                for f, fb_key in enumerate(full_base_keys):
                    if fb_key == base_key:
                        num_stars = max(num_stars, len(full_phrase_keys[f].split(' ')))
                sorted_keys[k] = key + ''.join([' ' for _ in range(num_stars)])

        # Add new entries to full dictionary
        for (k,s) in zip(sorted_keys, sorted_scores):
            #k = k.replace('>',u'\u2192')
            #k = k.replace('>','→')
            full_phrase_dict[k] = scores[s]

        # Choose color and update legend elements
        #color = all_colors[g]
        if 'Control' in group:
            color = 'red'
        else:
            color = 'forestgreen'
        color_dict[color] = sorted_keys
        if len(sorted_keys) > 0:
            legend_elements.append(Line2D([0], [0], color=color, lw=4, label=group))
    # Make wordcloud and recolor

    if len(legend_elements) == 0:
        print('No phrases!')
        return
      
    wordcloud = WordCloud(background_color='white', width=800, height=400, regexp=r"[a-zA-Z\u2192]+").generate_from_frequencies(full_phrase_dict)
    grouped_color_func = SimpleGroupedColorFunc(color_dict, 'grey')
    wordcloud.recolor(color_func = grouped_color_func)

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * .7, box.height])

    # Put a legend to the right of the current axis
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.,.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save
    plt.savefig(os.path.join(save_dir, 'wordcloud.pdf'))
    plt.close()
