#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : run_mcda.py
# @Author: nixin
# @Date  : 2021/11/26

import pandas as pd

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import minmax_scale
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from skcriteria import Data, MAX, MIN
from skcriteria.madm import simple, closeness
# import plotly.graph_objects as go
import numpy as np

solutions = pd.read_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/results (18).csv')
print(len(solutions))
print('==========')
# clean null soltuions
solutions = solutions[solutions['latent_inventive_solutions']!= '[]']
print(len(solutions))

count = pd.read_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/cleaned_count_patents.csv')

print(solutions.columns)
print(count.columns)

count = count[['patent_number', 'count_inventor_name', 'count_forward_cite_no_family',
                           'count_forward_cite_yes_family', 'count_backward_cite_no_family',
                           'count_backward_cite_yes_family']]

count = pd.merge(count,solutions[['patent_number', 'similarity_value']], on='patent_number')
count.to_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/mcda.csv', index = False)

print('=======')
print(count.columns)

## project the goodness for each column
criteria_data = Data(count.iloc[:, 1:7], [MAX, MAX, MAX, MAX,MAX,MAX],
                     anames= count['patent_number'],
                     cnames= count.columns[1:7],
                     weights= [0.1, 0.3, 0.1, 0.1, 0.1, 0.3]) ##assign weights to attributes
print(criteria_data)
print('++++++++')


print('==========')
dm = closeness.TOPSIS(mnorm="sum") # change the normalization criteria of the alternative matric to sum (divide every value by the sum opf their criteria)
dec = dm.decide(criteria_data)
print(dec)
print("Ideal:", dec.e_.ideal)
print("Anti-Ideal:", dec.e_.anti_ideal)
print("Closeness:", dec.e_.closeness) ##print each rank's value

count['rank_topsis'] = dec.e_.closeness
count = count.sort_values(by='rank_topsis', ascending=False)
print(count.columns)
print(count)
print(len(count))

rank = []
for i in range(len(count)):
    i = i+1
    rank.append(i)
print(rank)

count['rank'] = rank
print(count)
print(count.columns)
