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

# file_names = ["./model/simple_tagger_pos","./model/better_s_tagger_pos","./model/s2_tagger_pos","./model/s3_tagger_pos"]
# files = []
# for f in file_names:
#     j = JsonMetExtractor(f)
#     j.read()
#     files.append(j)
# for f,n in zip(files,file_names):
#     plt.plot(f.df['epoch_num'], f.df.validation_accuracy, label = n.split("/")[-1])
# plt.legend()
# plt.show()

# file_names = ["./model/simple_tagger_ner","./model/better_s_tagger_ner","./model/enc_only_s_tagger_ner","./model/emb_only_s_tagger_ner"]
# files = []
# for f in file_names:
#     j = JsonMetExtractor(f)
#     j.read()
#     files.append(j)
# for f,n in zip(files,file_names):
#     plt.plot(f.df['epoch_num'], f.df.validation_accuracy, label = n.split("/")[-1])
# plt.legend()
# plt.show()



file_names = ["./model/neural_crf_ner",
              "./model/better_neural_crf_ner",
              "./model/enc_only_neural_crf_ner",
              "./model/emb_only_neural_crf_ner"]
files = []
for f in file_names:
    j = JsonMetExtractor(f)
    j.read()
    files.append(j)

for f,n in zip(files,file_names):
    plt.plot(f.df['epoch_num'], f.df.validation_accuracy, label = n.split("/")[-1])

plt.legend()
plt.show()

sys.exit(1)
