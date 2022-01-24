# -*- coding: utf-8 -*-

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from App4api.bin import constants
from collections import OrderedDict
from App4api.bin.InformationExtractor import InformationExtractor
from App4api.bin.ParameterExtractor import ParameterExtractor
from App4api.bin.TechnologyFinder import TechnologyFinder

class ParamProcessor(object):

    def __init__(self, patents,input_folder, file_extension):
        self.patents = patents
        self.input_folder = input_folder
        self.file_extension = file_extension
        print("Processing started")

    def change_keys(self, dictionnary, number):
        number = number+'-'
        if type(dictionnary) is dict:
            return dict([(number+str(k) , self.change_keys(v, number)) for k, v in dictionnary.items()])
        else:
            return dictionnary

    def process_corpus(self):

        count_patent = 0
        patents = self.patents
        input_folder = self.input_folder
        project_folder = os.path.basename(os.path.normpath(input_folder))
        graph_folder = constants.GRAPH_FOLDER + project_folder+"/"
        output_result = []
        parameters_graph = []
        reduced_content = []
        patent_corpus = []
        source_list = []
        parameters_list =[]


        for patent_file in patents:

            read_patent = StringIO(patent_file)
            patent = json.load(read_patent)
            nNumber = patent['number']
            aAbstract = patent['abstract']
            cClaims = patent['claims']
            dDescription = patent['description']
            source = patent['source']

            patent_content = aAbstract + cClaims + dDescription
            patent_content = patent_content.splitlines()

            for line in patent_content:
                get_parameters = ParameterExtractor(line)
                parameters = get_parameters.extract_parameters()
                if parameters:
                    parameters_list.extend( parameters)


            parameters_list=list(set(parameters_list))

            parameters = dict(enumerate(parameters_list, 1))

            parameters = self.change_keys(parameters, nNumber.lower())

            parameters_array = OrderedDict({
                        "concept": {
                            "source": source,
                            "valeurs": parameters,

                        }

                    })
            pParameters= json.dumps(parameters_array, sort_keys=OrderedDict, indent=4, separators=(',', ': '))
            parameters_graph.append(pParameters)
            count_patent +=1
            source_list.append(source)
            patent_corpus.append(reduced_content)

        header = '{'
        parameters_output = '"parameters": [%s]' % ','.join(parameters_graph)
        footer = '}'
        output_result.extend((header, parameters_output,  footer))

        output_result = "".join(output_result)
        concepts_json = json.loads(output_result)


        json_write_to_file = json.dumps(concepts_json, sort_keys=False, indent=4, separators=(',', ': '))

        with open(graph_folder+"parameters-graph.json", 'w') as json_graph:
            json_graph.write(json_write_to_file)

        return concepts_json