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
from sklearn.metrics import accuracy_score
import numpy as np
from gensim.models import Word2Vec

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
        vectors.append(w2v(tokenizer(s)))

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



        

################################################################################
#   Main Method
#################################################################################



