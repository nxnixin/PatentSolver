#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
from io import StringIO
from App.bin import constants
import hashlib
from collections import OrderedDict
from App.bin.InformationExtractor import InformationExtractor
from App.bin.ParameterExtractor import ParameterExtractor
from App.bin.TechnologyFinder import TechnologyFinder
from App.bin.InformationExtractor_Claims import InformationExtractorClaims

class CorpusProcessor(object):

    def __init__(self, patents,input_folder, file_extension):
        self.patents = patents
        self.input_folder = input_folder
        self.file_extension = file_extension
        print("Processing started")


    def make_graphic (self, sizes, text, colors, labels):

        col = [[i / 255. for i in c] for c in colors]

        fig, ax = plt.subplots()
        ax.axis('equal')
        width = 0.35
        kwargs = dict(colors=col, startangle=180)
        outside, _ = ax.pie(sizes, radius=1, pctdistance=1 - width / 2, labels=labels, **kwargs)
        plt.setp(outside, width=width, edgecolor='white')

        kwargs = dict(size=20, fontweight='bold', va='center')
        ax.text(0, 0, text, ha='center', **kwargs)

        plt.show()

    def change_keys(self, dictionnary, number):
        number = number+'-'
        if type(dictionnary) is dict:
            return dict([(number+str(k) , self.change_keys(v, number)) for k, v in dictionnary.items()])
        else:
            return dictionnary

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
            output_json_claims ={}
            total_sentences_number_claims =0

            if type(patent_file) is dict:
                patent_file = json.dumps(patent_file)

            read_patent = StringIO(patent_file)
            patent = json.load(read_patent)
            nNumber = patent['number']
            aAbstract = patent['abstract']
            cClaims = patent['claims']
            dDescription = patent['description']

            root_img_url = 'https://worldwide.espacenet.com/espacenetImage.jpg?flavour=firstPageClipping&locale=en_EP&FT=D&'
            root_pdf_url = 'https://worldwide.espacenet.com/publicationDetails/originalDocument?'

            if nNumber is not None:
                match = re.search('(^[a-zA-Z]+)(([0-9]+)\s?([a-zA-Z0-9_]+$))', nNumber)
                # CC for country code
                CC = match.group(1)
                # NR for Number
                NR = match.group(2)
                NR = re.sub(r'\s', '', NR)
                # KC for Kind code
                KC = match.group(4)

                urlImg = root_img_url + '&CC=' + CC + '&NR=' + NR + '&KC=' + KC
                urlPDF = root_pdf_url + 'CC=' + CC + '&NR=' + NR + '&KC=' + KC + '&FT=D&ND=3&date=' + '&DB=&locale=en_EP#'



            #Find a more elegant way to do it
            patent_content = aAbstract + cClaims + dDescription
            patent_content = patent_content.splitlines()
            # for line in patent_content:
            #     line = self.dataCleaner(line)
            #     reduced_content.append(line)

            for line in patent_content:
                get_parameters = ParameterExtractor(line)
                parameters = get_parameters.extract_parameters()
                if parameters:
                    parameters_list.extend( parameters)
            for i in parameters_list:
                for j in parameters_list:
                    if i != j and len(i.split()) == 1:
                        if j.find(i) > -1 and i in parameters_list:

                            parameters_list.remove(i)

            parameters_list=list(set(parameters_list))
            if len(parameters_list) > 50:
                for i in parameters_list:
                    for j in parameters_list:
                        if i!=j:
                            comp = Levenshtein.ratio(i, j)
                            if comp >=.4 and i in parameters_list and j in parameters_list:
                                if len(i) > len(j):
                                    # print('{} is near duplicate of {}'.format(i, j))
                                    parameters_list.remove(i)

                for el in parameters_list:
                    if len(el.split()) == 1:
                        parameters_list.remove(el)

            parameters = dict(enumerate(parameters_list, 1))

            parameters = self.change_keys(parameters, nNumber.lower())



            source = input_folder+"/"+nNumber+file_extension.strip("*")

            parameters_array = OrderedDict({
                        "concept": {
                            "source": source,
                            "valeurs": parameters,
                            "image": urlImg,
                            "pdf": urlPDF
                        }

                    })
            pParameters= json.dumps(parameters_array, sort_keys=OrderedDict, indent=4, separators=(',', ': '))

            parameters_graph.append(pParameters)

            if dDescription !="" or cClaims!="":
                count_description +=1
                extract_concepts = InformationExtractor(dDescription,input_folder, file_extension, nNumber )
                output_json, total_sentences_number = extract_concepts.get_from_description()
                extract_concepts_claims = InformationExtractorClaims(cClaims,input_folder, file_extension, nNumber )
                output_json_claims_result= extract_concepts_claims.main()
                if output_json_claims_result is not None:
                    output_json_claims, total_sentences_number_claims = output_json_claims_result

                count_claims += 1
                if output_json is not None:
                    if type(output_json) is dict:
                        output_json = json.dumps(output_json)
                    extracted_concepts.append(output_json)
                    total_sentences_number += total_sentences_number
                if output_json_claims is not None :
                    if type(output_json_claims) is dict:
                        output_json_claims = json.dumps(output_json_claims)
                    extracted_concepts.append(output_json_claims)
                    total_sentences_number += total_sentences_number_claims
            elif cClaims !="":
                count_claims +=1
                print('Processing claims')
            else:
                count_abstract +=1
                print("processing abstract")
            count_patent +=1


            #print(source)
            source_list.append(source)
            patent_corpus.append(reduced_content)
        patent_corpus = dict(zip(source_list, patent_corpus))
        '''
        get_patent_technologies = TechnologyFinder(patent_corpus)
        technologies = get_patent_technologies.get_technologies()


        for source_file, technologies_list in technologies.items():

            technologies_array = OrderedDict({
                "concept": {
                    "source": source_file,
                    "values": technologies_list
                }

            })
            tTechnologies = json.dumps(technologies_array, sort_keys=OrderedDict, indent=4, separators=(',', ': '))

            technologies_graph.append(tTechnologies)
'''
        print(type(extracted_concepts))
        header = '{'
        graph = '"problem_graph": [%s],' % ','.join(extracted_concepts)
        parameters_output = '"parameters": [%s]' % ','.join(parameters_graph)
        #technologies_output = '"technologies": [%s]' % ','.join(technologies_graph)
        footer = '}'
        #output_result.extend((header, graph, parameters_output,technologies_output, footer ))
        output_result.extend((header, graph, parameters_output,  footer))

        output_result = "".join(output_result)
        output_result = re.sub(r'\,{2,}', ',', output_result)
        output_result = re.sub(r'\}\,\]', '}]', output_result)


        # exit()
        # print(output_result)
        concepts_json = json.loads(output_result)

        # concepts_json = json.loads(concepts_json)


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

        # original code
        with open(graph_folder+"graph.json", 'w') as json_graph:

        # with open(graph_folder + 'graph.json', 'w') as json_graph:
                json_graph.write(json_write_to_file)
        number_neutre = count_concepts - count_concepts_problem - count_concepts_solupart
        print("Le corpus contenait %s brevets dont %s abstract, %s revendications et %s descriptions" % (count_patent, count_abstract, count_claims, count_description))
        print("%s phrases ont été analysée(s)" % (total_sentences_number))
        print("%s concepts ont été trouvé(s) dont %s problèmes, %s solutions partielles et %s neutres" % (count_concepts, count_concepts_problem, count_concepts_solupart, number_neutre))

        #Display graphics
        first_color = (46, 204, 113)
        second_color = (245, 176, 65)
        #self.make_graphic([count_concepts_problem, count_concepts_solupart], "Ratio",[first_color,second_color],['Problems','Partial Solutions'])
        return json_write_to_file

    def process_corpus_json(self):

        count_abstract = 0
        count_claims = 0
        count_description = 0
        count_patent = 0
        total_sentences_number = 0
        count_concepts_solupart = 0
        count_concepts_problem = 0
        patents = self.patents
        input_folder = self.input_folder
        file_extension = self.file_extension
        project_folder = os.path.basename(os.path.normpath(input_folder))
        graph_folder = constants.GRAPH_FOLDER + project_folder + "/"
        extracted_concepts = []
        output_result = []
        parameters_graph = []
        reduced_content = []
        patent_corpus = []
        source_list = []
        parameters_list = []
        technologies_graph = []
        for patent_file in patents:
            # print(type(patent_file))

            #if type(patent_file) is dict:
            patent_file = json.dumps(patent_file)

            read_patent = StringIO(patent_file)
            patent = json.load(read_patent)
            # print(type(patent))
            filename = patent['filename']
            nNumber = patent['number']
            aAbstract = patent['abstract']
            cClaims = patent['claims']
            dDescription = patent['description']

            # Find a more elegant way to do it
            patent_content = aAbstract + cClaims + dDescription
            patent_content = patent_content.splitlines()
            # for line in patent_content:
            #     line = self.dataCleaner(line)
            #     reduced_content.append(line)

            for line in patent_content:
                get_parameters = ParameterExtractor(line)
                parameters = get_parameters.extract_parameters()
                if parameters:
                    parameters_list.extend(parameters)
            for i in parameters_list:
                for j in parameters_list:
                    if i != j and len(i.split()) == 1:
                        if j.find(i) > -1 and i in parameters_list:

                            parameters_list.remove(i)

            parameters_list = list(set(parameters_list))

            if len(parameters_list) > 50:
                for i in parameters_list:
                    for j in parameters_list:
                        if i!=j:
                            comp = Levenshtein.ratio(i, j)
                            if comp >=.4 and i in parameters_list and j in parameters_list:
                                if len(i) > len(j):
                                    # print('{} is near duplicate of {}'.format(i, j))
                                    parameters_list.remove(i)

                for el in parameters_list:
                    if len(el.split()) == 1:
                        parameters_list.remove(el)





            print('{} {}'.format('Taille: ', len(parameters_list)))


            parameters = dict(enumerate(parameters_list, 1))

            parameters = self.change_keys(parameters, nNumber.lower())

            source = input_folder + "/" + nNumber + file_extension.strip("*")

            parameters_array = OrderedDict({
                "concept": {
                    "source": source,
                    "valeurs": parameters
                }

            })
            pParameters = json.dumps(parameters_array, sort_keys=OrderedDict, indent=4, separators=(',', ': '))

            parameters_graph.append(pParameters)

            #if dDescription != "" and cClaims!="":
            if dDescription != "":
                count_description += 1
                extract_concepts = InformationExtractor(dDescription, input_folder, file_extension, filename)
                output_json, total_sentences_number_d = extract_concepts.get_from_description()
                if output_json != "":
                    extracted_concepts.append(output_json)
                total_sentences_number += total_sentences_number_d
                #count_claims += 1
                #extract_concepts = InformationExtractor(cClaims, input_folder, file_extension, nNumber)
                #output_json, total_sentences_number_c = extract_concepts.get_from_claims()
                #if output_json != "":
                    #extracted_concepts.append(output_json)
                #total_sentences_number_c += total_sentences_number_c
                #total_sentences_number = total_sentences_number_c+total_sentences_number_d

            elif cClaims != "":
                count_claims += 1
                extract_concepts = InformationExtractor(cClaims, input_folder, file_extension, nNumber)
                output_json, total_sentences_number = extract_concepts.get_from_claims()
                if output_json != "":
                    extracted_concepts.append(output_json)
                total_sentences_number += total_sentences_number
            elif dDescription != "":
                count_description += 1
                extract_concepts = InformationExtractor(dDescription, input_folder, file_extension, nNumber)
                output_json, total_sentences_number = extract_concepts.get_from_description()
                if output_json != "":
                    extracted_concepts.append(output_json)
                total_sentences_number += total_sentences_number
                count_claims += 1

            else:
                count_abstract += 1
                print("processing abstract")
            count_patent += 1

            # print(source)
        #     source_list.append(source)
        #     patent_corpus.append(reduced_content)
        # patent_corpus = dict(zip(source_list, patent_corpus))
        '''
        get_patent_technologies = TechnologyFinder(patent_corpus)
        technologies = get_patent_technologies.get_technologies()


        for source_file, technologies_list in technologies.items():

            technologies_array = OrderedDict({
                "concept": {
                    "source": source_file,
                    "values": technologies_list
                }

            })
            tTechnologies = json.dumps(technologies_array, sort_keys=OrderedDict, indent=4, separators=(',', ': '))

            technologies_graph.append(tTechnologies)
'''

        header = '{'
        graph = '"problem_graph": [%s],' % ','.join(extracted_concepts)
        parameters_output = '"parameters": [%s]' % ','.join(parameters_graph)
        # technologies_output = '"technologies": [%s]' % ','.join(technologies_graph)
        footer = '}'
        # output_result.extend((header, graph, parameters_output,technologies_output, footer ))
        output_result.extend((header, graph, parameters_output, footer))

        output_result = "".join(output_result)
        output_result = re.sub(r'\,{2,}', ',', output_result)
        output_result = re.sub(r'\}\,\]', '}]', output_result)
        concepts_json = json.loads(output_result)

        count_concepts = len(concepts_json['problem_graph'])
        for item, value in concepts_json.items():
            # if cle == "type" and value =="partialSolution":
            #   print ("yes")
            for element in value:
                for cle, valeur in element.items():
                    for k, v in valeur.items():
                        if k == "type" and v == "partialSolution":
                            count_concepts_solupart += 1
                        elif k == "type" and v == "problem":
                            count_concepts_problem += 1
        json_write_to_file = json.dumps(concepts_json, sort_keys=False, indent=4, separators=(',', ': '))
        # print(concepts_json.keys())
        with open(graph_folder + "graph.json", 'w') as json_graph:
            json_graph.write(json_write_to_file)

        print("Le corpus contenait %s brevets dont %s abstract, %s revendications et %s descriptions" % (
        count_patent, count_abstract, count_claims, count_description))
        print("%s phrases ont été analysée(s)" % (total_sentences_number))
        print("%s concepts ont été trouvé(s) dont %s problèmes et %s solutions partielles" % (
        count_concepts, count_concepts_problem, count_concepts_solupart))

        # Display graphics
        first_color = (46, 204, 113)
        second_color = (245, 176, 65)
        # self.make_graphic([count_concepts_problem, count_concepts_solupart], "Ratio",[first_color,second_color],['Problems','Partial Solutions'])
        return json_write_to_file