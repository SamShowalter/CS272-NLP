#################################################################################
#
#             Project Title:  Supervised Experiments              Class: CS272
#             Author:         Sam Showalter  
#             Date:           2021-04-04
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from preprocessing import Data 
from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics import accuracy_score, confusion_matrix

#################################################################################
#  Expiment functions for supervised classification
#################################################################################


def feature_ablation(data_name, featurization_list):
    """Try different feature generation techniques, including
    - CountVectorization
    - TFiDF
    - Stop words removal
    - Lemmatization


    """
    data = Data(data_name)
    perf_dict = {}
    for f, featurization in featurization_list.items():
        data.preprocess(featurization)

        clf = LogisticRegression()
        clf.fit(data.train_x, data.train_y)
        #Clear featurization so next run refreshes
        data.featurization = None

        # Predict
        preds = clf.predict(data.val_x)
        perf_dict[f] = accuracy_score(data.val_y, preds)

    return perf_dict
        



def dimensionality_ablation(data,num_comp_list):
    """Use truncated SVD to reduce dimensionality and
    see if that improves the performance of the classifier.

    - Normalize the input data (input features chosen as
     best from feature ablation).
    - Run truncated svd on normalized data, 
       saving only the top k components defined in input
    - Runs the default classifier (held constant)


    """
    perf_dict = {}
    for num_components in num_comp_list:
        train_x, train_y, val_x, val_y = data.train_x, data.train_y, data.val_x, data.val_y
        svd = TruncatedSVD(n_components = num_components)
        train_x = svd.fit_transform(train_x)
        val_x = svd.transform(val_x)

        clf = LogisticRegression()
        clf.fit(train_x, train_y)

        # Predict
        preds = clf.predict(val_x)
        perf_dict[num_components] = accuracy_score(val_y, preds)

    return perf_dict
        

def solver_pen_ablation(train_x, train_y, val_x, val_y, solvers, penalties):
    """Solvers and penalties to use when training 
    logistic regression classifier.
    
    :data: Training data defined from best of last two ablations
    :solvers: LR solver options
    :penalties: L1 and L2 penalty
    :returns: dictionary of performance (acc) scores across search

    """
    perf_dict = {"l1":{}, "l2":{}, 'none':{}}
    for solver in solvers:
        for penalty in penalties:
            try:
                clf = LogisticRegression(solver = solver, penalty = penalty)
                clf.fit(train_x, train_y)

                # Predict
                preds = clf.predict(val_x)
                perf_dict[penalty][solver] = accuracy_score(val_y, preds)
            except Exception as e:
                print(e)
                perf_dict[penalty][solver] = "-"

    return perf_dict
            




################################################################################
#   Main Method
#################################################################################

# Experiments run in the provided jupyter notebook 

