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
from speech import read_tsv
from icecream import ic
import tarfile
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize 
import string
#################################################################################
#   Function-Class Declaration
#################################################################################


class LemmaTokenizer(object):
    def __init__(self,tokenizer):
        self.wnl = WordNetLemmatizer()
        self.tokenizer = tokenizer
    def __call__(self, articles, punct = False):
        articles = self.tokenizer(articles)
        return [self.wnl.lemmatize(t).lower() for t in articles]

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
        


            


        
        


            
