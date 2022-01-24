
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File  : run.py
# @Author: nixin
# @Date  : 2021/11/26

import pandas as pd
from functions import *

from functools import partial
import multiprocessing as mp


df = pd.read_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/results (18).csv')
print(df.columns)

patent_number =[]
for patent in df['patent_number']:
    patent_number.append(patent)

print(patent_number)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~ Parameters for data_patent_details file ~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
path_to_data = "/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/"  #### don't forget to change

## Create csv file to store the data_patent_details from the patent runs
#  (1) Specify column order of patents
#  (2) Create csv if it does not exist in the data_patent_details path
data_column_order = ['inventor_name',
                     'assignee_name_orig',
                     'assignee_name_current',
                     'pub_date',
                     'priority_date',
                     'grant_date',
                     'filing_date',
                     'forward_cite_no_family',
                     'forward_cite_yes_family',
                     'backward_cite_no_family',
                     'backward_cite_yes_family',
                     'patent',
                     'url',
                     'abstract_text']

if 'edison_patents.csv' in os.listdir(path_to_data):
    os.remove(path_to_data + 'edison_patents.csv')  # delete previous csv file
    with open(path_to_data + 'edison_patents.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_column_order)
else:
    with open(path_to_data + 'edison_patents.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_column_order)
#
#
########### Run pool process #############
if __name__ == "__main__":
    ## Create lock to prevent collisions when processes try to write on same file
    l = mp.Lock()

    ## Use a pool of workers where the number of processes is equal to
    ##   the number of cpus - 1
    with poolcontext(processes=mp.cpu_count() - 1, initializer=init, initargs=(l,)) as pool:
        pool.map(partial(single_process_scraper, path_to_data_file=path_to_data + 'edison_patents.csv',
                         data_column_order=data_column_order),
                 patent_number)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~ clean raw data_patent_details ~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

##read Google scrawer's results
table = pd.read_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/edison_patents.csv')

# clean raw patent results
results = clean_patent(table)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~ count number ~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

results = count_patent(results)
print(results.columns)
results.to_csv('/Users/nixin/PycharmProjects/PatentSolver_demonstrator/MCDA/data/cleaned_count_patents.csv', index=False)
