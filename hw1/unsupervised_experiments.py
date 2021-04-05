#################################################################################
#
#             Project Title:  Unsupervised Experiments              Class: CS272
#             Author:         Sam Showalter
#             Date:           2021-04-04
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from preprocessing import Data, W2V_Vectorizer, SamsTokenizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score
import numpy as np
from gensim.models import Word2Vec
from speech import *
#################################################################################
#   Function-Class Declaration
#################################################################################
def w2v_build_dataset(w2v, tokenizer, data):
    """Word2vec dataset building
    :w2v: TODO
    :data): TODO
    :returns: TODO
    """
    vectors = []
    for s in data:
        v = tokenizer(s)
        if len(v) > 0:
            vectors.append(w2v(tokenizer(s)))
        else:
            vectors.append(np.zeros(300))

    return vectors

def w2v_ablation(models, fname,classifier, tokenizer = SamsTokenizer):
    """Word2Vec ablation to see if performance improves

    :model: TODO
    :data): TODO
    :returns: TODO

    """
    data = Data(fname)
    data.preprocess_labels(data.train_labels)
    data.preprocess_labels(data.val_labels)
    perf = []
    tokenizer = tokenizer()
    
    for model in models:
        model = Word2Vec.load("models/" + model)
        w2v = W2V_Vectorizer(model)
        data.train_x = np.array(w2v_build_dataset(w2v, tokenizer, data.train_data))
        data.val_x = np.array(w2v_build_dataset(w2v, tokenizer, data.val_data))

        # print(data.train_x.shape)
        # print(data.val_x.shape)


        clf = classifier()
        clf.fit(data.train_x, data.train_y)

        # Predict
        preds = clf.predict(data.val_x)
        perf.append(accuracy_score(data.val_y, preds))

    return perf



def w2v_ablation_all_features(models, fname,classifier, preprocessor, svd, tokenizer = SamsTokenizer):
    """Word2Vec ablation to see if performance improves + concatenation of old features

    :model: TODO
    :data): TODO
    :returns: TODO

    """
    data = Data(fname)
    data.preprocess(preprocessor, svd)
    perf = []

    tokenizer = tokenizer()
    
    for model in models:
        model = Word2Vec.load("models/" + model)
        w2v = W2V_Vectorizer(model)
        data.train_x_unlabeled = np.array(w2v_build_dataset(w2v, tokenizer, data.train_data))
        data.val_x_unlabeled = np.array(w2v_build_dataset(w2v, tokenizer, data.val_data))

        # print(data.train_x.shape)
        # print(data.val_x.shape)
        data.train_x_final = np.hstack((data.train_x, data.train_x_unlabeled))
        data.val_x_final = np.hstack((data.val_x, data.val_x_unlabeled))

        clf = classifier
        clf.fit(data.train_x_final, data.train_y)

        # Predict
        preds = clf.predict(data.val_x_final)
        perf.append(accuracy_score(data.val_y, preds))

    return perf
        

def read_unlabeled_w2v(tarfname, speech,model , tokenizer = SamsTokenizer()):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the speech.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    unlabeled.fnames = []
    for m in tar.getmembers():
            if "unlabeled" in m.name and ".txt" in m.name:
                    unlabeled.fnames.append(m.name)
                    unlabeled.data.append(read_instance(tar, m.name))
    unlabeled.X = speech.svd.transform(speech.featurization.transform(unlabeled.data))
    unlabeled.train_x_unlabeled = np.array(w2v_build_dataset(model, tokenizer, unlabeled.data))
    print(unlabeled.train_x_unlabeled.shape)
    # print(unlabeled.X.shape)
    unlabeled.X = np.hstack((unlabeled.X, unlabeled.train_x_unlabeled))
    tar.close()
    return unlabeled
################################################################################
#   Main Method
#################################################################################



