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
                 mets = ['epoch',
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
                jsn.append(json.loads(data))
        self.df = pd.DataFrame(jsn)
        print(self.df.sort_values('epoch'))




#################################################################################
#   Main Method
#################################################################################

# ext = JsonMetExtractor("./model/simple_tagger_pos")
# ext.read()

# plt.plot(ext.df['epoch'], ext.df.validation_accuracy)
# plt.plot(ext.df['epoch'], ext.df.training_accuracy)
# plt.show()

