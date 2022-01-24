
# ~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~~~ Import libraries ~~~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~ #

# Google Scraper Class #
from google_patent_scraper import scraper_class

# Context Manager #
from contextlib import contextmanager

# Writing/Reading
import csv
import numpy as np
import pandas as pd

# clean patent #
import re

# Multiprocessing #
import multiprocessing as mp

# parse xml to text
from bs4 import BeautifulSoup as bs

# zip folder to download
import shutil
import base64
import streamlit as st
import os

# extract problems
from App.bin import constants
from App.bin.InputHandler import InputHandler
from App.bin.PatentHandler import PatentHandler
from App.bin.CorpusProcessor import CorpusProcessor
import json
from pandas import json_normalize
import glob



# ~~~~~~~~~~~~~~~~~~~ #
# ~~~~ Functions ~~~~ #
# ~~~~~~~~~~~~~~~~~~~ #

def single_process_scraper(patent,path_to_data_file,data_column_order):
    """Scrapes a single google patent using the google scraper class
       
       Function does not return any values, instead it writes the output
         of the data_patent_details into a csv file specified in the path_to_data_file
         parameter

       Inputs:
         patent (str) : patent number including country prefix
         lock (obj) : to prevent collisions, function uses a lock. You can pass whichever
                      lock you want to this parameter
         path_to_data_file : absolute path to csv file to write data_patent_details to
         data_column_order : name of columns in order they will be saved in csv file

    """
    # ~ Initialize scraper class ~ #
    scraper=scraper_class() 

    # ~ Scrape single patent ~ #
    err, soup, url = scraper.request_single_patent(patent)

    # Checks if the scrape is successful.
    # If successful -> parse text and deposit into csv file
    # Else          -> print error statement

    if err=='Success':
        patent_parsed = scraper.get_scraped_data(soup,url,patent)

        # Save the parsed data_patent_details to a csv file
        #  using multiprocessing lock function
        #  to prevent collisions
        with lock:
            with open(path_to_data_file,'a',newline='') as ofile:
                writer = csv.DictWriter(ofile, fieldnames=data_column_order)
                writer.writerow(patent_parsed)
    else:
        print('Patent {0} has error code {1}'.format(patent,err))

# Allow pool to accept keyword arguments
@contextmanager
def poolcontext(*args, **kwargs):
    pool = mp.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def init(l):
    """Creates lock object that is global, for use in sharing 
       across processes
    """
    global lock
    lock = l


def patentinput(patent_string):
    """
    remove space among patent numbers from users' inputs
    """
    patent_string = patent_string.replace(" ", "") #remove space that user tpyed
    list_results = list(patent_string.split(","))
    return list_results

def clean_patent(table):
    """clean raw patent details from website
    """

    list_inventor_name = np.array([]) # create an empty list

    inventor_name = table['inventor_name']
    for line in inventor_name:
        new_line = re.sub(r'"inventor_name":', '', line)
        new_line = re.sub(r'\{|\}|\[|\]|\"', '', new_line)
        # print(new_line)
        list_inventor_name = np.append(list_inventor_name, new_line)

    new_table_inventor_name = pd.DataFrame(list_inventor_name, columns=['inventor_name'])
    # new_table.to_csv('saved_data/cleaned_patent_details')

    ##clean assignee_name_orig feature
    list_assignee_name = np.array([])
    assignee_name = table['assignee_name_orig']
    for line in assignee_name:
        new_line = re.sub(r'"assignee_name":', '', line)  ##### errors
        new_line = re.sub(r'\{|\}|\[|\]|\"', '', new_line)
        list_assignee_name = np.append(list_assignee_name, new_line)

    new_table_assignee_name = pd.DataFrame(list_assignee_name, columns=['assignee_name_orig'])
    # print(new_table_assignee_name)
    #
    ##clean assignee_name_current feature
    list_assignee_name_current = np.array([])
    assignee_name_current = table['assignee_name_current']
    for line in assignee_name_current:
        new_line = re.sub(r'("assignee_name":)|(\\n\s\s)|(\{|\}|\[|\]|\")', '', line)
        list_assignee_name_current = np.append(list_assignee_name_current, new_line)

    new_table_assignee_name_current = pd.DataFrame(list_assignee_name_current, columns=['assignee_name_current'])
    # print(new_table_assignee_name_current)
    #
    ##clean forward_cite_no_family feature
    list_forward_cite_no_family = np.array([])
    forward_cite_no_family = table['forward_cite_no_family']
    for line in forward_cite_no_family:
        new_line = re.sub(
            r'("patent_number":)|(\\n)|(\{|\}|\[|\]|\")|(priority_date)|(:)|(pub_date)|(\d{4}-\d{2}-\d{2})', '', line)
        new_line = re.sub(r'\s\,\s', '', new_line)
        list_forward_cite_no_family = np.append(list_forward_cite_no_family, new_line)

    new_table_forward_cite_no_family = pd.DataFrame(list_forward_cite_no_family, columns=['forward_cite_no_family'])
    # print(new_table_forward_cite_no_family)
    #
    ##clean forward_cite_yes_family feature
    list_forward_cite_yes_family = np.array([])
    forward_cite_yes_family = table['forward_cite_yes_family']
    for line in forward_cite_yes_family:
        new_line = re.sub(
            r'("patent_number":)|(\\n)|(\{|\}|\[|\]|\")|(priority_date)|(:)|(pub_date)|(\d{4}-\d{2}-\d{2})', '', line)
        new_line = re.sub(r'\s\,\s', '', new_line)
        list_forward_cite_yes_family = np.append(list_forward_cite_yes_family, new_line)

    new_table_forward_cite_yes_family = pd.DataFrame(list_forward_cite_yes_family, columns=['forward_cite_yes_family'])
    # print(new_table_forward_cite_yes_family)

    ##clean backward_cite_no_family feature
    list_backward_cite_no_family = np.array([])
    backward_cite_no_family = table['backward_cite_no_family']
    for line in backward_cite_no_family:
        new_line = re.sub(
            r'("patent_number":)|(\\n)|(\{|\}|\[|\]|\")|(priority_date)|(:)|(pub_date)|(\d{4}-\d{2}-\d{2})', '', line)
        new_line = re.sub(r'\s\,\s', '', new_line)
        list_backward_cite_no_family = np.append(list_backward_cite_no_family, new_line)

    new_table_backward_cite_no_family = pd.DataFrame(list_backward_cite_no_family, columns=['backward_cite_no_family'])
    # print(new_table_backward_cite_no_family)

    ##clean backward_cite_yes_family feature
    list_backward_cite_yes_family = np.array([])
    backward_cite_yes_family = table['backward_cite_yes_family']
    for line in backward_cite_yes_family:
        new_line = re.sub(
            r'("patent_number":)|(\\n)|(\{|\}|\[|\]|\")|(priority_date)|(:)|(pub_date)|(\d{4}-\d{2}-\d{2})', '', line)
        new_line = re.sub(r'\s\,\s', '', new_line)
        list_backward_cite_yes_family = np.append(list_backward_cite_yes_family, new_line)

    new_table_backward_cite_yes_family = pd.DataFrame(list_backward_cite_yes_family,
                                                      columns=['backward_cite_yes_family'])
    # print(new_table_backward_cite_yes_family)

    ##rename url feature
    list_patent_number = np.array([])
    patent_number = table['url']
    for line in patent_number:
        list_patent_number = np.append(list_patent_number, line)

    new_table_patent_number = pd.DataFrame(list_patent_number, columns=['patent_number'])
    # print(new_table_patent_number)

    ##rename patent feature
    list_patent_link = np.array([])
    patent_link = table['patent']
    for line in patent_link:
        list_patent_link = np.append(list_patent_link, line)

    new_table_patent_link = pd.DataFrame(list_patent_link, columns=['patent_link'])
    # print(new_table_patent_link)

    ##rename abstract_text
    list_abstract_text = np.array([])
    abstract_text = table['abstract_text']
    for line in abstract_text:
        list_abstract_text = np.append(list_abstract_text, line)

    new_table_abstract_text = pd.DataFrame(abstract_text, columns=['abstract_text'])
    # print(new_table_patent_link)

    ###################################

    ## concatenate all of sub dataframes to the final results
    results = pd.concat([new_table_patent_number, table[['pub_date', 'priority_date', 'grant_date', 'filing_date']],
                         new_table_inventor_name, new_table_assignee_name, new_table_assignee_name_current,
                         new_table_forward_cite_no_family, new_table_forward_cite_yes_family,
                         new_table_backward_cite_yes_family, new_table_backward_cite_no_family, new_table_patent_link,
                         new_table_abstract_text], axis=1)

    return results


def count_patent(patent_table):
    """count the patent features"""

    ##count the number of assignee_name feature
    assignee_name = pd.DataFrame(patent_table['assignee_name_orig'])
    count_assignee_name = assignee_name.applymap(lambda x: str.count(x, ',') + 1)
    count_assignee_name = count_assignee_name.rename(columns={'assignee_name_orig': 'count_assignee_name'})
    # print(count_assignee_name)

    ##count the number of inventor_name feature
    inventor_name = pd.DataFrame(patent_table['inventor_name'])
    count_inventor_name = inventor_name.applymap(lambda x: str.count(x, ',') + 1)
    count_inventor_name = count_inventor_name.rename(columns={'inventor_name': 'count_inventor_name'})
    # print(count_inventor_name)

    ##count the number of assignee_name_current feature
    assignee_name_current = pd.DataFrame(patent_table['assignee_name_current'])
    # print(assignee_name_current)

    ##replace NaN as int(0)
    assignee_name_current_replace_NaN = lambda x: int(0) if pd.isnull(x) else str.count(x, ',') + 1
    count_assignee_name_current = assignee_name_current.applymap(assignee_name_current_replace_NaN)
    count_assignee_name_current = count_assignee_name_current.rename(
        columns={'assignee_name_current': 'count_assignee_name_current'})
    # print(count_assignee_name_current)

    ##count forward_cite_no_family
    forward_cite_no_family = pd.DataFrame(patent_table['forward_cite_no_family'])
    forward_cite_no_family_replace_NaN = lambda x: int(0) if pd.isnull(x) else str.count(x, ',')
    count_forward_cite_no_family = forward_cite_no_family.applymap(forward_cite_no_family_replace_NaN)
    count_forward_cite_no_family = count_forward_cite_no_family.rename(
        columns={'forward_cite_no_family': 'count_forward_cite_no_family'})
    # print(count_forward_cite_no_family)

    ##count forward_cite_yes_family
    forward_cite_yes_family = pd.DataFrame(patent_table['forward_cite_yes_family'])
    forward_cite_yes_family_replace_NaN = lambda x: int(0) if pd.isnull(x) else str.count(x, ',')
    count_forward_cite_yes_family = forward_cite_yes_family.applymap(forward_cite_yes_family_replace_NaN)
    count_forward_cite_yes_family = count_forward_cite_yes_family.rename(
        columns={'forward_cite_yes_family': 'count_forward_cite_yes_family'})
    # print(count_forward_cite_yes_family)

    ##count backward_cite_no_family
    backward_cite_no_family = pd.DataFrame(patent_table['backward_cite_no_family'])
    backward_cite_no_family_replace_NaN = lambda x: int(0) if pd.isnull(x) else str.count(x, ',')
    count_backward_cite_no_family = backward_cite_no_family.applymap(backward_cite_no_family_replace_NaN)
    count_backward_cite_no_family = count_backward_cite_no_family.rename(
        columns={'backward_cite_no_family': 'count_backward_cite_no_family'})
    # print(count_backward_cite_no_family)

    ##count backward_cite_yes_family
    backward_cite_yes_family = pd.DataFrame(patent_table['backward_cite_yes_family'])
    backward_cite_yes_family_replace_NaN = lambda x: int(0) if pd.isnull(x) else str.count(x, ',')
    count_backward_cite_yes_family = backward_cite_yes_family.applymap(backward_cite_yes_family_replace_NaN)
    count_backward_cite_yes_family = count_backward_cite_yes_family.rename(
        columns={'backward_cite_yes_family': 'count_backward_cite_yes_family'})
    # print(count_backward_cite_yes_family)

    ##concate dataframes to the final cleaned dataset
    results = pd.concat([patent_table[['patent_number', 'pub_date', 'priority_date',
                                'grant_date', 'filing_date', 'inventor_name']], count_inventor_name,
                         patent_table[['assignee_name_orig']], count_assignee_name,
                         patent_table[['assignee_name_current']], count_assignee_name_current,
                         patent_table[['forward_cite_no_family']], count_forward_cite_no_family,
                         patent_table[['forward_cite_yes_family']], count_forward_cite_yes_family,
                         patent_table[['backward_cite_no_family']], count_backward_cite_no_family,
                         patent_table[['backward_cite_yes_family']], count_backward_cite_yes_family,
                         patent_table[['patent_link', 'abstract_text']]], axis=1)

    return results


def XMLtoTEXT(patent_xml, saved_file_path):
    # read file
    tree = bs(patent_xml, "html.parser")

    # get title

    print('Title:')
    title = tree.find_all("invention-title")
    patent_title = title[0].text
    print(patent_title)

    # get number
    print("Patent number:")
    patent_number = tree.find_all('doc-number')
    patent_number = 'US' + patent_number[0].text
    patent_number_new = re.sub(r'US0', 'US', patent_number)
    print(patent_number_new)

    # get domain
    print('Domain:')
    domain = tree.find_all('classification-level')
    patent_domain = domain[0].text
    print(patent_domain)

    # get date of publication
    print("Publication date:")
    date = tree.find_all("date")
    patent_pubdate = date[0].text
    print(patent_pubdate)

    # get abstract
    print('Abstract:')
    ab = tree.find_all("abstract")
    patent_abstract = ab[0].text
    print(patent_abstract)

    # get claim
    print('Claims:')
    claims = tree.find_all("claim-text")
    for claim in claims:
        print(claim.text)

    # get description
    print('Description:')
    description = tree.find_all('description')
    for des in description:
        print(des.text)

    # save file to the place
    with open(saved_file_path + patent_number_new + '.txt', 'w') as text_file:
        text_file.write("Patent title" + '\n' + patent_title +
                        '\n' * 2 + "Patent number" + '\n' +
                        patent_number_new + '\n' * 2 + "Domain" + '\n' + patent_domain + '\n' * 2 + "Publication date" + '\n' + patent_pubdate
                        + '\n' * 2 + "Abstract" + '\n' + patent_abstract
                        + '\n' * 2 + 'Claims' + '\n')  # save patent title, number, domain, publication data_patent_details, abstract
        for claim in claims:
            text_file.write(claim.text + '\n')
        text_file.write('\n' + 'Description' + '\n')
        for des in description:
            text_file.write('\n' + des.text + '\n')

    return text_file


# to download patents (.txt) by zip file
def create_download_zip(zip_directory, zip_path, filename):
    """
        zip_directory (str): path to directory you want to zip
        zip_path (str): where you want to save zip file
        filename (str): download filename for user who download this
    """
    shutil.make_archive(zip_path+filename, 'zip', zip_directory)

    with open(zip_path+filename+'.zip', 'rb') as f:
        st.download_button(
            label = 'Download',
            data = f,
            file_name='patent.zip',
            mime= 'zip'
        )



# save input files (txt) into the folder
def save_uploadedfile(uploadedfile):
    with open(os.path.join('Data/input/US_patents/',uploadedfile.name ), 'wb') as f:
        f.write(uploadedfile.getbuffer())
        # return st.success('Saved File:{}'.format(uploadedfile.name))

# to extract problems from patents
def extractor (folder):
    input_folder = constants.DATA_INPUT + folder
    files_extension = "*." + 'txt'

    iInput = InputHandler(input_folder, files_extension)
    input_data = iInput.get_input()

    pretreat_data = PatentHandler(input_data)
    clean_patent_data = pretreat_data.pretreat_data()

    process_data = CorpusProcessor(clean_patent_data, input_folder, files_extension)
    processed_data = process_data.process_corpus()

    # convert json to dataframe
    with open('Data/graphs/US_patents/graph.json') as json_data:
        data = json.load(json_data)

    concept_df = json_normalize(data['problem_graph'], sep="_")

    concept_df = concept_df[['concept_sentence', 'concept_source', 'concept_type']]
    problem_df = concept_df.rename(columns={"concept_sentence": "problem", 'concept_source': 'patent_number',
                                            'concept_type': 'type'})
    # choose problems
    problem_new = problem_df.loc[problem_df['type'] == 'problem']

    print(problem_new)

    new_table_test = problem_new['patent_number'].apply(
        lambda x: re.search(r'(?<=US_patents\/).*?(?=.txt)', x).group())

    # assign patent number to the corresponding feature
    problem_results = problem_new.assign(patent_number=new_table_test)

    print(problem_results[['problem', 'patent_number']])
    problem_results = problem_results[['patent_number', 'problem']]
    problem_results.to_csv('data_problem/problem.csv',
                           index=False)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def extract_info_text():
    new = pd.DataFrame(columns=['title', 'patent_number', 'domain', 'publication_date'])

    # use glob to get all the txt files in the folder
    path = 'Data/input/US_patents'
    txt_files = glob.glob(os.path.join(path, "*.txt"))
    for f in txt_files:
        df = pd.read_csv(f, sep='\n', header=None, names=['content'])
        print(df)
        # extract patent information from text
        new = new.append({'patent_number': df.iloc[3, 0], 'title': df.iloc[1, 0],
                          'domain': df.iloc[5, 0], 'publication_date': df.iloc[7, 0]}, ignore_index=True)

    print(new)

    problem = pd.read_csv('data_problem/problem.csv')
    final = pd.merge(problem, new, on='patent_number', how='left')
    return final

def input_domain(user_input_domain):
    if user_input_domain == 'A (Human necessities)':
        domain = 'A'
    elif user_input_domain == 'B (Performing operations; transporting)':
        domain = 'B'
    elif user_input_domain == 'C (Chemistry; metallurgy)':
        domain = 'C'
    elif user_input_domain == 'D (Textiles; paper)':
        domain = 'D'
    elif user_input_domain == 'E (Fixed constructions)':
        domain = 'E'
    elif user_input_domain == 'F (Mechanical engineering; lighting; heating; weapons; blasting engines or pumps':
        domain = 'F'
    elif user_input_domain == 'G (Physics)':
        domain = 'G'
    elif user_input_domain == 'H (Electricity)':
        domain = 'H'
    return domain

# the function for choosing month period that user choosed
def choosing_month_period(problem_corpus,start_year, end_year, start_month, end_month):
    problem_corpus = problem_corpus[problem_corpus['publication_year'].between(start_year, end_year)]
    if start_year != end_year:  # 2014- 2015 #2014- 2016
        if start_month == end_month:  # /01/  /01/
            if end_year == start_year + 1:  # 2014/03/01 - 2015/03/01 #2014/01/01 - 2015/01/23 #2014/12/01 - 2015/12/23
                problem_corpus.loc[(problem_corpus['publication_year'] == start_year) & (
                    problem_corpus['publication_month'].between(start_month, 12)), 'label'] = 'true'
                problem_corpus.loc[(problem_corpus['publication_year'] == end_year) & (
                    problem_corpus['publication_month'].between(1, end_month)), 'label'] = 'true'

            elif end_year > start_year + 1:  # 2014/01/01 - 2016/01/23 #2014/12/01 - 2016/12/23 # 2014/03/01 - 2016/03/01
                if start_month == 1:  # 2014/01/01 - 2016/01/23
                    problem_corpus.loc[(
                                               problem_corpus['publication_year'] == end_year) & (
                                       problem_corpus['publication_month'].between(
                                           end_month + 1, 12)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'
                elif start_month == 12:  # 2014/12/01 - 2016/12/23
                    problem_corpus.loc[(
                                               problem_corpus['publication_year'] == start_year) & (
                                       problem_corpus['publication_month'].between(
                                           1, start_month - 1)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'
                else:  # 2014/03/01 - 2016/03/01
                    problem_corpus.loc[(
                                               problem_corpus['publication_year'] == start_year) & (
                                       problem_corpus['publication_month'].between(
                                           1, start_month - 1)), 'label'] = 'false'
                    problem_corpus.loc[(
                                               problem_corpus['publication_year'] == end_year) & (
                                       problem_corpus['publication_month'].between(
                                           end_month + 1, 12)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'
        if start_month > end_month:  # /03/  /01/
            if end_year == start_year + 1:  # 2014/12/01 - 2015/03/01 #2014/02/01 - 2015/01/23
                problem_corpus.loc[(problem_corpus['publication_year'] == start_year) & (
                    problem_corpus['publication_month'].between(start_month, 12)), 'label'] = 'true'
                problem_corpus.loc[(problem_corpus['publication_year'] == end_year) & (
                    problem_corpus['publication_month'].between(1, end_month)), 'label'] = 'true'

            elif end_year > start_year + 1:  # 2014/12/01 - 2016/03/01 #2014/02/01 - 2016/01/23
                problem_corpus.loc[(
                                           problem_corpus['publication_year'] == start_year) & (
                                   problem_corpus['publication_month'].between(
                                       1, start_month - 1)), 'label'] = 'false'
                problem_corpus.loc[(
                                           problem_corpus['publication_year'] == end_year) & (
                                   problem_corpus['publication_month'].between(
                                       end_month + 1, 12)), 'label'] = 'false'
                problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'

        if start_month < end_month:  # /01/  /03/
            if end_year == start_year + 1:  # 2014/01/01 - 2015/12/01 #2014/02/01 - 2015/11/23
                problem_corpus.loc[(problem_corpus['publication_year'] == start_year) & (
                    problem_corpus['publication_month'].between(start_month, 12)), 'label'] = 'true'
                problem_corpus.loc[(problem_corpus['publication_year'] == end_year) & (
                    problem_corpus['publication_month'].between(1, end_month)), 'label'] = 'true'

            elif end_year > start_year + 1:  # 2014/01/01 - 2016/12/01 #2014/02/01 - 2016/11/23
                if start_month == 1 & end_month == 12:  # 2014/01/01 - 2016/12/01
                    problem_corpus['label'] = 'true'
                elif start_month == 1:  # 2014/01/01 - 2016/03/01 #2014/01/01 - 2016/11/01
                    problem_corpus.loc[(problem_corpus['publication_year'] == end_year) & (problem_corpus[
                                                                                               'publication_month'].between(
                        end_month + 1, 12)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'
                elif end_month == 12:  # 2014/02/01 - 2016/12/01 #2015/02/01 - 2016/12/01
                    problem_corpus.loc[(problem_corpus['publication_year'] == start_year) & (problem_corpus[
                                                                                                 'publication_month'].between(
                        1, start_month - 1)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'
                else:  # 2014/02/01 - 2016/11/23
                    problem_corpus.loc[(problem_corpus['publication_year'] == start_year) & (problem_corpus[
                                                                                                 'publication_month'].between(
                        1, start_month - 1)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus['publication_year'] == end_year) & (problem_corpus[
                                                                                               'publication_month'].between(
                        end_month + 1, 12)), 'label'] = 'false'
                    problem_corpus.loc[(problem_corpus.label != 'false'), 'label'] = 'true'



    else:  # start_year == end_year: 2012-2012
        problem_corpus = problem_corpus[problem_corpus['publication_year'] == start_year]
        if start_month != end_month:  # 2014/03/01 - 2014/05/01 2014/01/01 - 2014/05/01 2014/03/01 - 2014/12/01
            problem_corpus.loc[problem_corpus['publication_month'].between(start_month, end_month), 'label'] = 'true'
        else:  # 2014/03/01 - 2014/03/20 #2014/01/01 - 2014/01/20
            problem_corpus.loc[problem_corpus['publication_month'] == start_month, 'label'] = 'true'

    problem_corpus = problem_corpus.loc[problem_corpus['label'] == 'true']
    problem_corpus= problem_corpus[['patent_number', 'Domain', 'First part Contradiction',
                                     'Second part Contradiction', 'publication_date', 'publication_year',
                                     'publication_month', 'label']]
    return problem_corpus

# for IDM-Similar model (word2vec)
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

def creat_query_id(dataset):
    # create query
    question = []
    for each in dataset['problem']:
        new = "What is the solution for the problem that " + each + "?"
        question.append(new)
    dataset['question'] = question

    # create id
    data = dataset.rename(columns={'Unnamed: 0': 'id'})
    return data

def csv_to_json (csv_file,json_file):
    results = []
    with open(csv_file) as csv_file:
        csvReader = csv.DictReader(csv_file)
        for row in csvReader:
            context = row['Context']
            qas = []
            content = {}
            content['id'] = row['id']
            content['question'] = row['question']
            qas.append(content)
            result = {}
            result['context'] = context
            result['qas'] = qas
            results.append(result)

    # write data to a json file
    with open(json_file, 'w') as jsonFile:
        jsonFile.write(json.dumps(results, indent=4))



def QA_prediction(prediction_file, prediction_output, model):
    # if __name__ == '__main__':
    with open(prediction_file, 'r') as pre_file:
        temp = json.loads(pre_file.read())
        predictions = model.predict(temp)

    with open(prediction_output, 'w') as json_file:
        json_file.write(json.dumps(predictions, indent=4))
    print(predictions)

def json_to_csv(input_file, output_file):
    result = pd.read_json(input_file)
    print(result.head())

    result_answer = result.iloc[0][:]
    print(result_answer.head())
    print(len(result_answer))

    df = pd.DataFrame(index=np.arange(len(result_answer)), columns=['id', 'answer'])
    print(df)

    for i in range(len(result_answer)):
        line = result_answer[i]
        print(line)
        df.iloc[i, 0] = line['id']
        df.iloc[i, 1] = line['answer']

    print(df.head())
    df.to_csv(output_file, index=False)
