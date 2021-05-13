#################################################################################
#
#             Project Title:  Extract NLP Metrics              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-05-05
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from glob import glob
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
#################################################################################
#   Function-Class Declaration
################################################################################

class JsonMetExtractor(object):

    """Extract data from allenNLP logs"""

    def __init__(self,
                 path,
                 mets = ['epoch_num',
                         'training_accuracy',
                         'validation_accuracy',
                         'training_loss',
                         'validation_loss',
                         ]):
        """Prepare to extract information
        from checkpoints of allennlp model"""
        self.path = path
        self.mets = mets

    def read(self):
        """Read in all data from JSON into pandas df

        Saves info as pandas df

        """
        files = glob(self.path + "/*epoch*.json")
        jsn = []
        for f in files:
            with open(f, 'rb') as file:
                data = file.read()
                jsn.append(flatten_json(json.loads(data)))
        self.df = pd.DataFrame(jsn).sort_values('epoch_num')
        # print(self.df.sort_values('epoch_num'))

def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


#################################################################################
#   Main Method
#################################################################################

prefix = 'model/'

s_pos_p = '_s_tagger_pos'
s_ner_p = '_s_tagger_ner'
n_pos_p = '_neural_crf_pos'
n_ner_p = '_neural_crf_ner'

dirs = os.listdir(prefix)

##Toggle these for plots
#keyword = n_ner_p
#file_names = [prefix + d for d in dirs if keyword in d]

#files = []
#for f in file_names:
#    j = JsonMetExtractor(f)
#    j.read()
#    files.append(j)
#for f,n in zip(files,file_names):
#    label =n.split("/")[-1].replace(keyword,"")
#    if label == "":
#        label = 'baseline'
#    print(label)
#    plt.plot(f.df['epoch_num'], f.df.validation_accuracy, label =label)
#plt.legend()
#plt.title("Validation Accuracy of Neural Tagger for NER", fontsize = 15)
#plt.ylabel("Validation Accuracy", fontsize = 12)
#plt.xlabel("Epoch Number", fontsize = 12)
#plt.show()
#sys.exit(1)

#######################################################################
# Cross best plots

# cross_dirs = [prefix + d for d in dirs if d.endswith('ner') and
#               (d.startswith("_") or 'gru' in d)]
# labels = ['gru_neural','gru_simple','baseline_simple','baseline_neural']
# print(cross_dirs)

# files = []
# for f in cross_dirs:
#     j = JsonMetExtractor(f)
#     j.read()
#     files.append(j)

# for i,(f,n) in enumerate(zip(files,cross_dirs)):
#     plt.plot(f.df['epoch_num'], f.df.validation_accuracy, label =labels[i])
# plt.legend()
# plt.title("Validation Accuracy for NER performance", fontsize = 15)
# plt.ylabel("Validation Accuracy", fontsize = 12)
# plt.xlabel("Epoch Number", fontsize = 12)
# plt.show()
# sys.exit(1)

#######################################################################
# Cross best from baseline to best (GRU)
baselines = ['_s_tagger_ner','_s_tagger_pos','_neural_crf_ner','_neural_crf_pos']
best = ['gru_s_tagger_ner','gru_s_tagger_pos','gru_neural_crf_ner','gru_neural_crf_pos']
titles =['simple_NER','simple_POS','neural_NER','neural_POS']

bases = []
for b in baselines:
    j = JsonMetExtractor(prefix + b)
    j.read()
    bases.append(j.df)

bests = []
for b in best:
    j = JsonMetExtractor(prefix + b)
    j.read()
    bests.append(j.df)

for ba,be,title in zip(bases,bests,titles):
    col_pref ='validation_accuracy_per_label_'
    cols =[i for i in be.columns if i.startswith('validation_accuracy_per_label_')]
    ba = ba[['epoch_num'] + cols]
    be = be[['epoch_num'] + cols]
    diffs = sorted([(c.replace('validation_accuracy_per_label_',''), abs(ba[c].iloc[-1] - be[c].iloc[-1]))
                    for c in cols], reverse = True,key=lambda tup:tup[1])
    cols_to_plot = [tup[0] for tup in diffs[:8]]
    print(cols_to_plot)

    for c in cols_to_plot:
        plt.plot(ba['epoch_num'],be[col_pref + c] - ba[col_pref + c],label = c)
    plt.legend()
    plt.title("Performance Difference, Baseline v. Best for {}".format(title), fontsize = 15)
    plt.ylabel("Validation Accuracy Difference", fontsize = 12)
    plt.xlabel("Epoch Number", fontsize = 12)
    plt.show()

#######################################################################
# TODO: NOW we just need it for test data








