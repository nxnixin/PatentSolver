#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : test_sentence_similarity.py
# @Author: nixin
# @Date  : 2019-03-06

import numpy as np
from scipy import spatial
from gensim.models import word2vec
import pandas as pd



# load the trained word vector model
model = word2vec.Word2Vec.load('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/Word2vec/trained_word2vec.model')
index2word_set = set(model.wv.index2word)

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

#read problem file
problem_corpus = pd.read_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/Word2vec/data_problem_corpus/problem_corpus_sample_cleaned.csv')
problem_corpus = problem_corpus.head(100)

target_problem = 'strategic cleavage of such a target rna will destroy its ability to direct synthesis of an encoded protein'
target_domain = 'A'

# remove the same domain's problems
problem_corpus = problem_corpus[problem_corpus.Domain != 'A']


# choose the time range
problem_corpus = problem_corpus[problem_corpus['publication_year'].between(2015, 2017)]


value=[]
for each_problem in problem_corpus['First part Contradiction']:
    s1_afv = avg_feature_vector(target_problem, model=model, num_features=100, index2word_set=index2word_set)
    s2_afv = avg_feature_vector(each_problem, model=model, num_features=100, index2word_set=index2word_set)
    sim_value = format( 1 - spatial.distance.cosine(s1_afv, s2_afv), '.2f')
    value.append(sim_value)

problem_corpus[['similarity_value', 'target_problem']] = value, target_problem

print(problem_corpus)

# set similarity threshold
problem_corpus_final = problem_corpus[problem_corpus.similarity_value>= '0.8']
# print(problem_corpus.columns())

problem_corpus_final.to_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/Word2vec/simialrity_result/test.csv', index=False)
print(problem_corpus_final)






