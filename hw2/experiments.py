#################################################################################
#
#             Project Title:  Experiments, hw2              Class: CS272
#             Author:         Samuel Showalter
#             Date:           2021-04-20
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from data import *
from lm import *
from generator import Sampler
import pickle as pkl
from tqdm import tqdm

#################################################################################
#   Function-Class Declaration
#################################################################################

def addk_ablation(n,ks):
    """Add-k smoothing ablation
    Will use bigram model

    :n: n-gram cardinality
    :ks: list of k-coefficients

    """
    res_dict = {}


    # Do no run, the following function was used to generate the splits
    # file_splitter("data/reuters.txt")
    dnames = ["brown", "reuters", "gutenberg"]
    for d in dnames:
        res_dict[d] = {}

    data_dict = {}
    for d in dnames:
        data_dict[d] =read_texts("data/corpora.tar.gz", d)

    # Unigram model information
    for d in dnames:
        print("Unigram: {}".format(d))
        data = data_dict[d]
        model = learn_unigram(data)
        res_dict[d][1] = {}
        res_dict[d][1]['perplexity'] = {}
        res_dict[d][1]['perplexity']['train'] = model.perplexity(data.train)
        res_dict[d][1]['perplexity']['dev'] = model.perplexity(data.dev)
        res_dict[d][1]['perplexity']['test'] = model.perplexity(data.test)


    # Learn the models for each of the domains, and evaluate it
    for d in dnames:
        data = data_dict[d]
        res_dict[d][n] = {}
        for k in tqdm(ks):
            res_dict[d][n][k] = {}
            res_dict[d][n][k]['perplexity'] = {}
            model = learn_ngram(data,n, k= k, smoothing = 'add-k')
            res_dict[d][n][k]['perplexity']['train'] = model.perplexity(data.train)
            res_dict[d][n][k]['perplexity']['dev'] = model.perplexity(data.dev)
            res_dict[d][n][k]['perplexity']['test'] = model.perplexity(data.test)

    with open("data/results/addk_ablation.pkl", 'wb') as file:
        pkl.dump(res_dict,file)

    return res_dict

def backoff_ablation(ns,k, res_dict):
    """TODO: Docstring for backoff_ablation.

    :n: n-gram cardinalities to try
    :res_dict: add to original dictionary
    :returns: TODO

    """
    dnames = ["brown", "reuters", "gutenberg"]
    for d in dnames:
        if d not in res_dict:
            res_dict[d] = {}

    data_dict = {}
    for d in dnames:
        data_dict[d] =read_texts("data/corpora.tar.gz", d)

    # Learn the models for each of the domains, and evaluate it
    for d in dnames:
        print(d)
        data = data_dict[d]
        for n in tqdm(ns):
            if n not in res_dict[d]:
                res_dict[d][n] = {}
            res_dict[d][n]["backoff"] = {}
            res_dict[d][n]["backoff"]['perplexity'] = {}
            model = learn_ngram(data,n,k = None, smoothing = 'backoff')
            res_dict[d][n]["backoff"]['perplexity']['train'] = model.perplexity(data.train)
            res_dict[d][n]["backoff"]['perplexity']['dev'] = model.perplexity(data.dev)
            res_dict[d][n]["backoff"]['perplexity']['test'] = model.perplexity(data.test)

    with open("data/results/backoff_ablation.pkl", 'wb') as file:
        pkl.dump(res_dict,file)
    return res_dict


################################################################################
#   Main Method
#################################################################################



