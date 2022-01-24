from App.bin.FiguresCleaner import FiguresCleaner
from App.bin.ParameterExtractor import ParameterExtractor
from App.bin import constants
import nltk
import re
import os

import json
import hashlib
import Levenshtein
import uuid
from collections import OrderedDict
from App.bin.SharpClassifier import SharpClassifier
from App.bin.ClassifierWithIncr import ClassifyWithIncr_it


class InformationExtractorClaims(object):

    def __init__(self, section, input_folder, file_extension, file_name):
        self.section = section
        self.input_folder = input_folder
        self.file_extension = file_extension
        self.file_name = file_name

        patent_abbreviations = open(constants.ASSETS + "abbreviation_sentence_splitter").read().split()
        sentence_finder = nltk.data.load('tokenizers/punkt/english.pickle')
        sentence_finder._params.abbrev_types.update(patent_abbreviations)
        self.sentence_finder = sentence_finder

    def clean_data (self, sentence):

        sentence = str(sentence.lower())
        sentence = re.sub(r'\(\s,?\s?\)', '', sentence)
        sentence = re.sub(r'\s+,', ',', sentence)
        sentence = re.sub(r'^\d+', '', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        if sentence is not None:
            return sentence

    def truncate_data (self, sentence):

        sentence = str(sentence.lower())
        sentence = re.sub(r'wherein said\s*', '', sentence)
        sentence = re.sub(r'characterized in that said\s*|characterised in that said?\s*', '', sentence)
        sentence = re.sub(r'wherein\s*', '', sentence)
        sentence = re.sub(r'characterized\s*|characterised\s*', '', sentence)
        sentence = re.sub(r'characterized in that\s*', '', sentence)
        sentence = re.sub(r'where\s*', '', sentence)
        sentence = re.sub(r'where said\s*', '', sentence)
        sentence = re.sub(r'further comprising', 'the system or method comprises', sentence)
        sentence = re.sub(r'.*thereof\s*\,?', '', sentence)
        sentence = re.sub(r'^\s+', '', sentence)
        sentence = re.sub(r'\s+\.$', '', sentence)
        if sentence is not None:
            return sentence

    def selectLines(self, line, lexic):
        with open(constants.ASSETS + lexic) as n:
            inclusion_list = n.read().splitlines()
            claims_words = re.compile('|'.join(inclusion_list))
            m = re.search(claims_words, line)
            if m is not None:
                return m.group(1)
                # pass
            # return line
    def main(self):

        output_result = []
        compt_Id = 50
        count_concept = 3

        clean_content_list = []
        concept_list = []

        output_content = []

        uniq_output_linked_content =[]
        parameters_list = []
        total_sentences_number =0
        section = self.section
        input_folder = self.input_folder
        file_name = self.file_name
        file_extension = self.file_extension
        projectFolder = os.path.basename(os.path.normpath(input_folder))
        output_file_name = input_folder+"/"+file_name+file_extension.strip("*")

        root_img_url = 'https://worldwide.espacenet.com/espacenetImage.jpg?flavour=firstPageClipping&locale=en_EP&FT=D&'
        root_pdf_url = 'https://worldwide.espacenet.com/publicationDetails/originalDocument?'



        if file_name is not None:
            match = re.search('(^[a-zA-Z]+)(([0-9]+)\s?([a-zA-Z0-9_]+$))', file_name)
            # CC for country code
            CC = match.group(1)
            #NR for Number
            NR = match.group(2)
            NR = re.sub(r'\s', '', NR)
            #KC for Kind code
            KC = match.group(4)

            urlImg = root_img_url+'&CC='+CC+'&NR='+NR+'&KC='+KC
            urlPDF = root_pdf_url+'CC='+CC+'&NR='+NR+'&KC='+KC+'&FT=D&ND=3&date='+'&DB=&locale=en_EP#'

        graphItemId = hashlib.md5(file_name.encode())
        graphItemIdValue = graphItemId.hexdigest()
        graphItemIdValue = str(uuid.uuid4())

        sentence_finder = self.sentence_finder
        sentences = sentence_finder.tokenize(section.strip())
        for sentence in sentences:
            # print(sentence)
            sentence = self.clean_data(sentence)
            if sentence !='':
                clean_content_list.append(sentence)
        for line in clean_content_list:
            # print(len(line.split()))
            if not re.match(r'^\s*$', line):

                line = self.selectLines(line, 'claims_indices')

                if line is not None and count_concept > 0:
                    line = self.truncate_data(line)
                    line = re.sub(r'in that', '', line)
                    # print(line, len(line.split()))
                    concept_list.append(line)
                    count_concept -= 1

        count_concept = 3
        if len(concept_list) is not None:
            total_sentences_number = len(concept_list)
            for concept in concept_list :


                if concept is not None and not re.match(r'^\s,', concept) and len(concept.split())<50:
                    classifyT = ClassifyWithIncr_it()
                    polarite = classifyT.main(concept)
                    get_parameters = ParameterExtractor(concept)
                    parameters = get_parameters.extract_parameters()

                    parameters_list.extend(parameters)

                    values = OrderedDict({
                        "concept": {
                            "type": polarite,
                            "id": graphItemIdValue + str(compt_Id),
                            "sentence": concept,
                            "source": output_file_name,
                            "parameters": parameters_list,
                            "image": urlImg,
                            "pdf": urlPDF

                        }

                    })
                    json_string = json.dumps(values, sort_keys=OrderedDict, indent=4, separators=(',', ': '))
                    output_result.append(json_string)
                    output_result = list(set(output_result))

                output_json = ",".join(output_result)

                return output_json, total_sentences_number



