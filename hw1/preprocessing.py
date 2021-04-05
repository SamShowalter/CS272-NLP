#################################################################################
#
#             Project Title:  Preprocessing hw1              Class: CS272  
#             Author:         Sam Showalter 
#             Date:           2021-04-01
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from sklearn.preprocessing import LabelEncoder , Normalizer 
from sklearn.feature_extraction.text import CountVectorizer
from speech import read_tsv
from icecream import ic
import tarfile
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import string
import numpy as np
#################################################################################
#   Function-Class Declaration
#################################################################################

class W2V_Vectorizer(object):

    """Word 2 Vec Feature creation"""

    def __init__(self, model, dim = 100):
        """Use Word2Vec Model trained on unlabeled data
        to increase performance of model.

        :model: TODO

        """
        self._dim = dim
        self._model = model

    def _embed(self, token):
        """embed tokens with model

        :token: TODO
        :returns: TODO

        """
        if token in self._model:
            return self._model[token]
        return np.zeros(self._dim)

    def __call__(self, token_vec):
        """
        Return mean embedding (sum would be uneven because 
        of variable length input).

        This may hide some of the more important stuff but its
        one idea
        """
        tok_vecs = [self._embed(token) for token in token_vec]
        return np.array(tok_vecs).mean(axis = 0)

        


class SamsTokenizer():
    # Stopwords handled by word2vec system
    def __init__(self, lower = True, punct = False):
        self.lower = lower
        self.punct = punct
        self.wnl = WordNetLemmatizer()
        self.tokenizer = word_tokenize
    def __call__(self, articles, punct = False):
        articles = self.tokenizer(articles.decode('utf-8'))
        tokens = [self.wnl.lemmatize(t) for t in articles]
        if self.lower:
            tokens = [self.wnl.lemmatize(t).lower() for t in articles]
        if not self.punct:
            tokens = [t for t in tokens if t not in string.punctuation]

        return tokens

class Data():

    """House for preprocessing all the data"""

    def __init__(self, filename):
        """CS272 hw1 data prep

        :filename): TODO

        """
        self.featurization = None
        # self.vectorizer = TfidfVectorizer
        self.norm = Normalizer()
        tar = tarfile.open("data/" + filename, "r:gz")
        self.count_vect = None
        self.le = None
        self.svd = None
        self.filename = filename
        ic("-- train data")
        self.train_data, self.train_fnames, self.train_labels = read_tsv(tar, "train.tsv")
        ic(len(self.train_data)) 
        ic("-- val data") 
        self.val_data, self.val_fnames, self.val_labels = read_tsv(tar, "dev.tsv")
        ic(len(self.val_data))

        # self.preprocess()
        # tar.close()

    def preprocess(self, featurization, svd = None, norm = False):
        self.preprocess_features(self.train_data, featurization, svd, norm)
        self.preprocess_features(self.val_data, featurization, svd, norm)
        self.preprocess_labels(self.train_labels)
        self.preprocess_labels(self.val_labels)
    
    def preprocess_features(self, data, featurization,svd, norm = False):
        """Preprocess data for train and dev sections
        :train: TODO
        :returns: TODO

        """
        if not self.featurization:

            self.featurization = featurization
            self.train_x = self.featurization.fit_transform(data)
            if norm:
                self.train_x = self.norm.fit_transform(self.train_x)
            if svd:
                self.svd = svd
                self.train_x = self.svd.fit_transform(self.train_x)
            # print(self.count_vect.vocabulary_) 
        else:
            self.val_x = self.featurization.transform(data)
            if norm:
                self.val_x = self.norm.transform(self.val_x)

            if self.svd:
                self.val_x = self.svd.transform(self.val_x)



    def preprocess_labels(self,labels):
        """Label encoder
        :returns: TODO

        """
        if not self.le:
            self.le = LabelEncoder()
            self.le.fit(labels)
            self.target_labels = self.le.classes_
            self.train_y = self.le.transform(self.train_labels)

        else:
            self.val_y = self.le.transform(self.val_labels)
        


            


        
        


            
