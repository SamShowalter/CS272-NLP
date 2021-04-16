#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import numpy as np
import re
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        numOOV = self.get_num_oov(corpus)
        return pow(2.0, self.entropy(corpus, numOOV))

    def get_num_oov(self, corpus):
        vocab_set = set(self.vocab())
        words_set = set(itertools.chain(*corpus))
        numOOV = len(words_set - vocab_set)
        return numOOV

    def tokenize(self, sentence):
        """Tokenize input sentence

        :sentence: TODO
        :returns: TODO

        """
        no_extra_spaces_lower = re.sub("\s\s+"," ",sentence).lower()
        alpha_numeric = re.sub("[^0-9a-zA-Z ]","", no_extra_spaces_lower)
        return alpha_numeric.split(" ")

    def entropy(self, corpus, numOOV):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s, numOOV)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence, numOOV):
        p = 0.0
        sentence = ['<SOS>'] + self.tokenize(sentence) + ['<EOS>']
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i], numOOV)
        # p += self.cond_logprob('END_OF_SENTENCE', sentence, numOOV)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, unk_prob=0.0001):
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        sentence = ['<SOS>'] + self.tokenize(sentence) + ['<EOS>']
        for w in sentence:
            self.inc_word(w)

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous, numOOV):
        if word in self.model:
            return self.model[word]
        else:
            return self.lunk_prob-log(numOOV, 2)

    def vocab(self):
        return self.model.keys()


# TODO: Work on
# YOU NEED TO COMPLETE THIS
class Ngram(LangModel):
    def __init__(self, ngram_size, unk_prob = 0.0001):
        self.n = ngram_size
        self.context_size = self.n-1
        self.lunk_prob = log(unk_prob, 2)
        self.model = {}
        self.vocab = set()


    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence):
        tokens = self.tokenize(sentence) + ['<EOS>']

        # Handle start of sentence or where not enough tokens


        # Trigram example
        # [this, is, a, token] -> (this is) -> a
        for i in range(self.n - 1, len(tokens), 1):
            context = tuple(sentence[i-self.context_size:i])
            self.add_or_inc(context, sentence[i])


    def add_or_inc(self, context, word):
        """Add or increment either context or
        model-context dictionary

        :context: TODO
        :returns: TODO

        """
        #Increment context
        self.vocab.add(word)
        if (context not in self.model):
            self.model[context] = {}
            self.model[context]['context_size'] = 1
        else:
            self.model[context]['context_size'] += 1

        #Increment context
        if (word not in self.model[context]):
            self.model[context][word] = 1
        else:
            self.model[context][word] += 1

    # optional, if there are any post-training steps (such as normalizing probabilities)
    # I handle this in different functions
    # I also can calculate probabilities in O(1) time so not really needed
    def norm(self): pass

    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV):
        if (previous in self.model) and (word in self.vocab):
            return np.log2((self.model[previous][word])
                    / self.model['context_size'])

        #If word not in vocab but context is
        elif (previous in self.model):
            return 1 / self.model[previous]['context_size']

        # If word is not part of the original vocabulary
        # and neither is the context
        else:
            return self.lunk_prob

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self.vocab
