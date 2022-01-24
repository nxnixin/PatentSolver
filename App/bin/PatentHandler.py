# -*- coding: utf-8 -*-

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port 8080
import glob
import nltk
import os
import re
import codecs
import chardet
import shutil
import json
from io import StringIO
from App.bin import constants
from App.bin.FiguresCleaner import FiguresCleaner


from collections import OrderedDict

class PatentHandler(object):

    def __init__(self, patents):
        self.patents = patents

    def custom_cleaner(self, line):
        line = str(line)
        #line = line.lower()
        line = re.sub(r'PatentInspiration Url', '', line)
        line = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', line)
        line = re.sub(r'{', '(', line)
        line = re.sub(r'&quot;', '\'', line)
        line = re.sub(r'}', ')', line)
        line = re.sub(r'\t.*patentinspiration.*\n', '', line)
        line = re.sub(r'^|\n{2,}\bAbstract\b\n?', '', line)
        line = re.sub(r'^|\n{2,}\bClaims\b\n?', '', line)
        line = re.sub(r'^|\n{2,}\bDescription\b\n?', '', line)
        line = re.sub(r'fig\.', 'figure', line)
        line = re.sub(r'Fig\.', 'Figure', line)
        line = re.sub(r'FIG\.', 'Figure', line)
        line = re.sub(r'figs\.', 'figures', line)
        line = re.sub(r'FIGS\.', 'Figures', line)
        line = re.sub(r'(\w+\.)', r'\1 ', line)
        line = re.sub(r'&#39;', '\'', line)
        line = re.sub(r'&gt;', '>', line)
        line = re.sub(r'&lt;', '<', line)
        line = re.sub(r'&#176;', ' deg.', line)
        line = re.sub(r'  ', ' ', line)
        line = line.strip()
        return line

    def dataCleaner(self,line):
        with open(constants.ASSETS + "dropPart") as l:
            # next(l)
            drop_part = l.read().splitlines()
            drop_part_pattern = re.compile('|'.join(drop_part))

        line = str(line)
        #line = line.lower()
        line = re.sub(r'^([A-Z-/]+\s)+([A-Z])', r'\n\2', line)
        line = re.sub(drop_part_pattern, r'\n', line)
        line = re.sub(r'\s+\.\s?\d+\s+', ' ', line)
        line = line.strip()
        return line

    def smooth_data_cleaner(self,line):
        line = str(line)
        # line = line.lower()
        line = re.sub(r'\s+,', ',', line)
        line = re.sub(r'\d\w-\d\w (and? \d\w-\d\w)?', '', line)
        line = re.sub(r'\d\w-\d\w', '', line)
        line = re.sub(r'\(\s?(,\s?|;\s?)+\s?\)', '', line)
        line = re.sub(r'\s+\.\s\.', '.\n', line)
        line = re.sub(r'\s+\.\s+([a-z]+)', r' \1', line)
        line = re.sub(r'\s+(\.)\s+\[\s?\d+\s?]\s+', r'.\n', line)
        line = re.sub(r'\s?\[\s?\d+\s?]\s+', r'\n', line)
        line = re.sub(r'\s+(\.)\s+([A-Z]+)', r'.\n\2', line)
        line = re.sub(r'\s+;\s+', '; ', line)
        line = re.sub(r'\(\s+\'\s+\)', '', line)
        line = re.sub(r'\(\s+\)', '', line)
        line = re.sub(r'\(\s?\.\s?\)', '', line)
        line = re.sub(r'\(\s/\s?\)', '', line)
        line = re.sub(r'\s{2,}', ' ', line)
        line = re.sub(r'(\d+)\s+(\.)\s+(\d+)', r'\1.\3', line)
        line = line.strip()
        return line


    def get_project_folder(self):
        patents = self.patents
        if patents:
            file = patents[0]
            project_folder = os.path.basename(os.path.dirname(file))
            return project_folder

    def convert_to_uf8(self, input_file_name,output_file_name, file_encoding):

        BLOCKSIZE = 1048576
        with codecs.open(input_file_name, "r", file_encoding) as input_file:
            with codecs.open(output_file_name, "w", "utf-8") as output_file:
                while True:
                    file_contents = input_file.read(BLOCKSIZE)
                    if not file_contents:
                        break
                    output_file.write(file_contents)

    def sectionFinder(self, file_name, start_delimiter, end_delimiter):

        patent_file = open(file_name, encoding='utf-8')
        section = ""
        found = False

        for line in patent_file:
            if found :
                section += line
                if line.strip() == end_delimiter:
                    break
            else:
                if line.strip() == start_delimiter:
                    found = True
                    # abstract = "Abstract\n"
        return section

    def pretreat_data(self):
        clean_patent_data= []
        patents = self.patents

        project_folder = self.get_project_folder()

        # original code
        # corpus_folder = constants.CORPUS + project_folder + "/"

        corpus_folder = str(constants.CORPUS)+str(project_folder)+"/"
        temp_folder = str(constants.TEMP)+str(project_folder)+"/"
        graph_folder = str(constants.GRAPH_FOLDER)+str(project_folder)+"/"

        folders = [corpus_folder, temp_folder, graph_folder]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
            else:
                shutil.rmtree(folder)
                os.makedirs(folder)

        for patent in patents:

            patent_name_with_extension = os.path.basename(patent)
            patent_name, extension= patent_name_with_extension.split('.')
            corpus_patent_path = corpus_folder + patent_name_with_extension
            #temp_patent_path = temp_folder + patent_name+'.json'

            patent_binary = open(patent, 'rb').read()

            file_encoding = chardet.detect(patent_binary)
            file_encoding = file_encoding['encoding']
            self.convert_to_uf8(patent,corpus_patent_path, file_encoding)

            temp_file = StringIO()
            #print(temp_patent_path)
            a_abstract = self.sectionFinder(corpus_patent_path,"Abstract", "Claims")
            a_abstract= self.custom_cleaner(a_abstract)
            abstract_cleaner = FiguresCleaner(a_abstract)
            a_abstract = ''.join(abstract_cleaner.clean_figures())
            a_abstract = self.smooth_data_cleaner(a_abstract)
            a_abstract = self.dataCleaner(a_abstract)

            c_claims = self.sectionFinder(corpus_patent_path, "Claims", "")
            c_claims = self.custom_cleaner(c_claims)
            claims_cleaner = FiguresCleaner(c_claims)
            c_claims = ''.join(claims_cleaner.clean_figures())
            c_claims = self.smooth_data_cleaner(c_claims)
            c_claims = self.smooth_data_cleaner(c_claims)

            d_description = self.sectionFinder(corpus_patent_path,"Description", "Claims")
            d_description = self.custom_cleaner(d_description)
            description_cleaner = FiguresCleaner(d_description)
            d_description = ''.join(description_cleaner.clean_figures())
            d_description = self.smooth_data_cleaner(d_description)
            d_description = self.dataCleaner(d_description)

    #TODO Manipulate data on system memory.

            data = {

                'number': patent_name,
                'abstract': a_abstract,
                'claims': c_claims,
                'description': d_description
            }

            json.dump(data, temp_file)
            clean_patent_data.append(temp_file.getvalue())
        return clean_patent_data


    def pretreat_json(self):
        clean_patent_data= []
        patents = self.patents
        temp_file = StringIO()

        for patent in patents:
            patent = json.dumps(patent)

            read_patent_t = StringIO(patent)
            patent_section = json.load(read_patent_t)
            filename = patent_section['filename']
            number = patent_section['number']

            a_abstract = patent_section['abstract']
            a_abstract= self.custom_cleaner(a_abstract)
            abstract_cleaner = FiguresCleaner(a_abstract)
            a_abstract = ''.join(abstract_cleaner.clean_figures())
            a_abstract = self.smooth_data_cleaner(a_abstract)
            a_abstract = self.dataCleaner(a_abstract)

            c_claims = patent_section['claims']
            c_claims = self.custom_cleaner(c_claims)
            claims_cleaner = FiguresCleaner(c_claims)
            c_claims = ''.join(claims_cleaner.clean_figures())
            c_claims = self.smooth_data_cleaner(c_claims)
            c_claims = self.smooth_data_cleaner(c_claims)

            d_description = patent_section['description']
            d_description = self.custom_cleaner(d_description)
            description_cleaner = FiguresCleaner(d_description)
            d_description = ''.join(description_cleaner.clean_figures())
            d_description = self.smooth_data_cleaner(d_description)
            d_description = self.dataCleaner(d_description)

    #TODO Manipulate data on system memory.

            data = {
                'filename': filename,
                'number': number,
                'abstract': a_abstract,
                'claims': c_claims,
                'description': d_description
            }


            clean_patent_data.append(data)
            #json.dumps(clean_patent_data, temp_file)

        #print(json.dumps(clean_patent_data))
        return clean_patent_data











