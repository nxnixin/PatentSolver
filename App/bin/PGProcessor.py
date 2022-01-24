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

class PGProcessor(object):

    def __init__(self, patents,input_folder, file_extension):
        self.patents = patents
        self.input_folder = input_folder
        self.file_extension = file_extension
        print("Processing started")

    def process_corpus(self):

        count_abstract = 0
        count_claims = 0
        count_description = 0
        count_patent = 0
        total_sentences_number =0
        count_concepts_solupart = 0
        count_concepts_problem = 0
        patents = self.patents
        input_folder = self.input_folder
        file_extension = self.file_extension
        project_folder = os.path.basename(os.path.normpath(input_folder))
        graph_folder = constants.GRAPH_FOLDER + project_folder+"/"
        extracted_concepts = []
        output_result = []
        parameters_graph = []
        reduced_content = []
        patent_corpus = []
        source_list = []
        parameters_list =[]
        technologies_graph =[]


        for patent_file in patents:

            read_patent = StringIO(patent_file)
            patent = json.load(read_patent)
            nNumber = patent['number']
            aAbstract = patent['abstract']
            cClaims = patent['claims']
            dDescription = patent['description']
            source = patent['source']

            if dDescription !="":
                count_description +=1
                extract_concepts = InformationExtractor(dDescription,input_folder, file_extension, nNumber, source )
                output_json, total_sentences_number = extract_concepts.get_from_description()
                if output_json !="":
                    extracted_concepts.append(output_json)
                total_sentences_number += total_sentences_number
            elif cClaims !="":
                count_claims +=1
                print('Processing claims')
            else:
                count_abstract +=1
                print("processing abstract")
            count_patent +=1


            #print(source)
            source_list.append(source)


        header = '{'
        graph = '"problem_graph": [%s]' % ','.join(extracted_concepts)
        footer = '}'
        output_result.extend((header, graph, footer))
        output_result = "".join(output_result)
        concepts_json = json.loads(output_result)
        count_concepts = len(concepts_json['problem_graph'])
        for item, value in concepts_json.items():
            #if cle == "type" and value =="partialSolution":
             #   print ("yes")
            for element in value:
                for cle, valeur in element.items():
                    for k,v in valeur.items():
                        if k == "type" and v =="partialSolution":
                            count_concepts_solupart += 1
                        elif k == "type" and v =="problem":
                            count_concepts_problem += 1
        json_write_to_file = json.dumps(concepts_json, sort_keys=False, indent=4, separators=(',', ': '))
        #print(concepts_json.keys())
        with open(graph_folder+"graph.json", 'w') as json_graph:
            json_graph.write(json_write_to_file)

        print("Le corpus contenait %s brevets dont %s abstract, %s revendications et %s descriptions" % (count_patent, count_abstract, count_claims, count_description))
        print("%s phrases ont été analysée(s)" % (total_sentences_number))
        print("%s concepts ont été trouvé(s) dont %s problèmes et %s solutions partielles" % (count_concepts, count_concepts_problem, count_concepts_solupart))

        #Display graphics
        first_color = (46, 204, 113)
        second_color = (245, 176, 65)
        #self.make_graphic([count_concepts_problem, count_concepts_solupart], "Ratio",[first_color,second_color],['Problems','Partial Solutions'])
        return concepts_json