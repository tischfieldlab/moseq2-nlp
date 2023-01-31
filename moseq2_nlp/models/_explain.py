from lime.lime_text import LimeTextExplainer
import re
import pdb

class Explainer(object):

    def __init__(self, feature_map, classifier, class_names,
                 bow=True,
                 bad_syllables = [-5],
                 num_transitions = 70,
                 max_syllable = 70):

        self.clf = classifier
        self.explainer = LimeTextExplainer(class_names=class_names, bow=bow)
        self.feature_map = feature_map

        self.feature_map_kwargs = {'bad_syllables': bad_syllables,
                                   'num_transitions': num_transitions,
                                   'max_syllable': max_syllable}

    def predict(self, sentences):
        pdb.set_trace()
        formatted_sentences = [re.sub('\  +', ' ', sent).split(' ') for sent in sentences]
        for s, sentence in enumerate(formatted_sentences):

            if sentence[0] == '':
                formatted_sentences[s] = formatted_sentences[s][1:]
            if sentence[-1] == '':
                formatted_sentences[s] = formatted_sentences[s][:-1]

        features = self.feature_map(formatted_sentences, **self.feature_map_kwargs)
        return self.clf.predict_proba(features)

    def explain_instance(self, sentence, num_samples=100):
        return self.explainer.explain_instance(sentence, self.predict, num_samples=num_samples)
    
    def explain_class():
        raise NotImplemented
