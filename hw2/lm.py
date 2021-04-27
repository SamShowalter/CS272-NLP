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

        # Language model nodes
        # The -> over whole vocab
        # The -> {"dog": 2,"cat": 4} -- man, woman
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
        # print(sentence)
        if isinstance(sentence,list):
            sentence = " ".join(sentence)
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
        sentence = sentence + ['END_OF_SENTENCE']
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
        self.context_size = 0
        self.n = 1
        self.model = dict()
        self.lunk_prob = log(unk_prob, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        # print(self.tokenize(sentence))
        # sentence = self.tokenize(sentence)
        sentence = sentence + ['END_OF_SENTENCE']
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


class Ngram(LangModel):
    def __init__(self, ngram_size, smoothing = 'add-k', k = 0.01, unk_prob = 0.0001):
        self.n = ngram_size
        self.smoothing = smoothing
        self.k = k
        self.context_size = self.n-1
        self.lunk_prob = log(unk_prob, 2)
        self.model = {}
        self.vocabulary = set()

        if self.smoothing == 'backoff':
            self.backoff_models = {}
            for i in range(1, self.n + 1):
                self.backoff_models[i] = {}

    def make_valid_prefix(self, sentence):
        if sentence == False:
            sentence = []

        l_s = len(sentence)

        if l_s < self.context_size:
            sentence = ['<SOS>'] + sentence
            if (l_s < self.context_size - 1):
                padding_num = self.context_size - l_s - 1
                sentence = ['<padding>']*padding_num + sentence

        valid_prefix = tuple(sentence[-self.context_size:])
        return valid_prefix

    def fit_sentence(self, sentence):
        if self.smoothing == 'backoff':
            self._fit_sentence_unigram(sentence)
            for i in range(2, self.n + 1):
                self._fit_sentence_ngram(sentence, i)
        else:
            self._fit_sentence_ngram(sentence, self.n)

    def _fit_sentence_unigram(self, sentence):
        sentence = sentence + ['END_OF_SENTENCE']
        for w in sentence:
            self.add_or_inc(None, w, 1)

    def _fit_sentence_ngram(self, sentence, n):

        #Pad start of every sentence so all characters modeled
        # Context size is always greater than zero
        context_size = n - 1
        pref =['<SOS>']
        padding = []
        if (context_size -1) > 0:
            padding = ['<padding>']*(context_size - 1)
        pref = padding + pref

        # sentence = tokenize(sentence)
        tokens = pref + sentence + ['END_OF_SENTENCE']

        # Trigram example
        # [this, is, a, token] -> (this is) -> a
        for i in range(n - 1, len(tokens), 1):
            context = tuple(tokens[i-context_size:i])
            self.add_or_inc(context, tokens[i], n)


    def add_or_inc(self, context, word, n):
        """Add or increment either context or
        model-context dictionary

        :context: TODO
        :returns: TODO

        """
        #Increment context

        if n == self.n:
            model = self.model
        else:
            model = self.backoff_models[n]

        #Add word to vocabulary
        self.vocabulary.add(word)

        if (n == 1):
            self._add_inc_unigram(word,model)
        else:
            self._add_inc_ngram(context, word, model)

    def _add_inc_unigram(self,w, model):
        if w in model:
            model[w] += 1.0
        else:
            model[w] = 1.0

    def _add_inc_ngram(self, context, word, model):

        if (context not in model):
            model[context] = {}
            model[context]['context_size'] = 1

        else:
            model[context]['context_size'] += 1

        #Increment context
        if (word not in model[context]):
            model[context][word] = 1
        else:
            model[context][word] += 1

    def unigram_norm(self, prob_model, model):
        """Normalize to probabilities"""
        tot = 0.0
        for word in model:
            tot += model[word]
        for word in model:
            prob_model[word] = model[word]/tot

    def ngram_norm(self, prob_model, model):
        #Model for ngram normalization
        vocab_len = len(self.vocabulary)
        for context in model.keys():
            prob_model[context] = {}
            if self.k == None:
                self.k = 0
            prob_model[context]['<OOV>'] = (self.k /
                        (model[context]['context_size'] + self.k*vocab_len))
            for word in model[context].keys():
                if (word in ['context_size','<OOV>']):
                    continue
                prob_model[context][word] = \
                    ((model[context][word] + self.k) /
                     (model[context]['context_size'] + self.k*vocab_len))

    # optional, if there are any post-training steps (such as normalizing probabilities)
    # I handle this in different functions
    # I also can calculate probabilities in O(1) time so not really needed
    def norm(self):
        """
        Make conditional probability distribution
        for all of the contexts

        """
        self.prob_model = {}

        if self.smoothing == 'backoff':
            self.backoff_prob_models = {}
            self.backoff_prob_models[1] = {}
            self.unigram_norm(self.backoff_prob_models[1],
                              self.backoff_models[1])
            for i in range(2, self.n+1):
                self.backoff_prob_models[i] = {}
                self.ngram_norm(self.backoff_prob_models[i] if i != self.n else self.prob_model,
                                self.backoff_models[i] if i != self.n else self.model)
        else:
            self.ngram_norm(self.prob_model,
                            self.model)


    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous, numOOV):
        # Three cases
        # - Context exists and token exists within it
        # - Context exists but token does not
        # - Neither exists

        previous = self.make_valid_prefix(previous)
        if self.smoothing != 'backoff':
            return self.add_k_cond_logprob(word, previous)

        else:
            return self.backoff_cond_logprob(word, previous)

    def backoff_cond_logprob(self, word,previous, lmbda = 0.4):
        """Backoff conditional probabilities

        :word: TODO
        :previous: TODO
        :returns: TODO

        """
        n = self.n
        prob_model = self.prob_model
        mult = 1

        #Try ngrams
        while (n >1):
            if (previous in prob_model) and (word in prob_model[previous]):
                return np.log2(mult*prob_model[previous][word])
            else:
                mult *= lmbda
                n -=1
                previous = previous[1:]
                prob_model = self.backoff_prob_models[n]

        unigrams = self.backoff_prob_models[1]

        #Try unigrams
        if word in unigrams:
            return np.log2(mult*unigrams[word])

        #Return lunk prob
        return self.lunk_prob



    def add_k_cond_logprob(self, word, previous):
        """TODO: Docstring for add_k_cond_prob.

        :word: TODO
        :previous: TODO
        :returns: TODO

        """
        if (previous in self.prob_model) and (word in self.prob_model[previous]):
            # print( ((self.prob_model[previous][word])
            #         / self.prob_model[previous]['context_size']))
            return np.log2(self.prob_model[previous][word])


        #If word not in vocab but context is
        elif ((previous) in self.prob_model):
            return np.log2(self.prob_model[previous]['<OOV>'])

        # If word is not part of the original vocabulary
        # and neither is the context
        else:
            return self.lunk_prob

    # required, the list of words the language model suports (including EOS)
    def vocab(self):
        return self.vocabulary


