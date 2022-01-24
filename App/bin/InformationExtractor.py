# -*- coding: utf-8 -*-

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port 8080
import nltk
nltk.download('all')
import os
import re
import json
import hashlib
import Levenshtein
import uuid
from App.bin import constants
from collections import OrderedDict
from nltk import word_tokenize

from App.bin.SharpClassifier import SharpClassifier
from App.bin.ClassifierWithIncr import ClassifyWithIncr_it
from App.bin.SentenceClassifier import SentenceClassifier
from App.bin.ParameterExtractor import ParameterExtractor

class InformationExtractor(object):

    patent_abbreviations = open(constants.ASSETS + "abbreviation_sentence_splitter").read().split()
    sentence_finder = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_finder._params.abbrev_types.update(patent_abbreviations)

    def __init__(self, section, input_folder,file_extension, file_name):
        self.section = section
        self.input_folder = input_folder
        self.file_extension = file_extension
        self.file_name = file_name

        print("Extracting problem graph")

    #@staticmethod


    def discardLines(self, line,lexic):
        with open (constants.ASSETS+ lexic) as m:
            exclusion_list = m.read().splitlines()
            if any(word in line for word in exclusion_list):
                pass
            else:
                return line


    def selectLines(self, line, lexic):
        with open(constants.ASSETS + lexic) as n:
            inclusion_list = n.read().splitlines()
            if any(word in line for word in inclusion_list):
                return line

    def last_cleansing(self, concept):
        concept = str(concept)
        concept = concept.lower()
        if concept.endswith("."):
            concept = concept.strip(".")
        concept = re.sub(r'^consequently ','', concept)
        concept = re.sub(r'^such ', '', concept)
        concept = re.sub(r'^said ', '', concept)
        concept = re.sub(r'^\s+', '', concept)
        concept = re.sub(r'^it is worth noting that ', '', concept)
        concept = re.sub(r'^example of ', '', concept)
        concept = re.sub(r'^since ', '', concept)
        concept = re.sub(r'^\( |\)$ ', '', concept)
        return concept

    # def get_from_claims(self):
    #
    #     section = self.section
    #     content = []
    #     sentence_finder = InformationExtractor.sentence_finder
    #     sentences = sentence_finder.tokenize(section.strip())
    #     with open(constants.ASSETS + "getFromClaims") as concept:
    #         # next(concept)
    #         included_words = concept.read().splitlines()
    #         include_link_pattern = re.compile('|'.join(included_words))


    def get_from_description(self):
        previous_polarity = ''
        noise_trash =[]

        content = []
        include_links = []
        output_content = []
        ex_output_content = []
        output_result=[]
        output_linked_content = []
        output_inter_content = []
        uniq_output_linked_content =[]
        ex_output_content_linked =[]
        section = self.section
        input_folder = self.input_folder
        file_name = self.file_name
        file_extension = self.file_extension
        projectFolder = os.path.basename(os.path.normpath(input_folder))
        output_file_name = input_folder+"/"+file_name+file_extension.strip("*")

        graphItemId = hashlib.md5(file_name.encode())
        graphItemIdValue = graphItemId.hexdigest()
        graphItemIdValue = str(uuid.uuid4())
        t_sline = ""
        t_sline_ex =[]
        compt_Id = 30
        compt_Id_ex = 40

        root_img_url = 'https://worldwide.espacenet.com/espacenetImage.jpg?flavour=firstPageClipping&locale=en_EP&FT=D&'
        root_pdf_url = 'https://worldwide.espacenet.com/publicationDetails/originalDocument?'

        if file_name is not None:
            match = re.search('(^[a-zA-Z]+)(([0-9]+)\s?([a-zA-Z0-9_]+$))', file_name)
            # CC for country code
            CC = match.group(1)
            # NR for Number
            NR = match.group(2)
            NR = re.sub(r'\s', '', NR)
            # KC for Kind code
            KC = match.group(4)

            urlImg = root_img_url + '&CC=' + CC + '&NR=' + NR + '&KC=' + KC
            urlPDF = root_pdf_url + 'CC=' + CC + '&NR=' + NR + '&KC=' + KC + '&FT=D&ND=3&date=' + '&DB=&locale=en_EP#'

        sentence_finder = InformationExtractor.sentence_finder

        #section = self.dataCleaner(section)
        #print(section)
        sentences = sentence_finder.tokenize(section.strip())


        with open(constants.ASSETS + "includeLinks") as concept:
            # next(concept)
            included_words = concept.read().splitlines()
            include_link_pattern = re.compile('|'.join(included_words))
        #open examplification wordfile
        with open(constants.ASSETS + "examplificationclues") as examplif:
            # next(concept)
            exam_words = examplif.read().splitlines()
            examplif_word_pattern = re.compile('|'.join(exam_words))

        description_sentences_number = len(sentences)
        number_of_words = 0
        for sentence in sentences:

            # with open(constants.DATA + 'sentences.txt', 'a', encoding='utf8') as file_handler:
            #     for item in sentences:
            #         file_handler.write("{}\n".format(item))
            number_of_word = len(nltk.word_tokenize(sentence))
            number_of_words += number_of_word


            sentenced = self.discardLines(sentence, "exclusionList")


            if sentenced is not None:


                content.append(sentenced)
                #print("origine=> "+sentence)
        total_sentences_number = len(sentences)
        # mean_sentence_length = int(round(number_of_words/total_sentences_number))
        # print(mean_sentence_length)

        for line in content:

            line = self.selectLines(line, "inclusionList")



            if line is not None:

                if re.match(include_link_pattern, line):
                    include_links.append(line)
                    #print(line)
                if line.count(',') == 0:
                    output_content.append(line)
                    # content.remove(line)
                if line.count(',') > 0:
                    output_inter_content.append(line)
                    content.remove(line)
        for s in content:
            # print(s, file_name)
            sentence = self.discardLines(s, "FilterS")
            if sentence is not None:
                if s.count(',') <= 2 and re.match(examplif_word_pattern, s.lower()):
                    s = str(s)
                    cs = s.lower()
                    cs = re.sub(examplif_word_pattern, '', cs)
                    cs = re.sub('which', 'this/these', cs)
                    cs = re.sub(r'\.$', '', cs)
                    #print(s)
                    if cs.count(',') == 1 and cs.count('such as')==0:
                        ex_output_content_linked.append(cs)
                    else:
                        ex_output_content.append(cs)
                elif s.count(',') == 1:
                    s = str(s)
                    s = s.lower()
                    s = self.selectLines(s, "OneCommaDiscriminator")
                    if s is not None:
                    #s = re.sub('which', 'this/these', s)
                    #print(s)
                        s = re.sub(r'^thus, ', '', s)
                        s = re.sub(r'^preferably, ', '', s)
                        s = re.sub(r'^conventional ', '', s)
                        s = re.sub(r'^in particular, ', '', s)
                        s = re.sub(r'^specifically, ', '', s)
                        s = re.sub(r'^as necessary, ', '', s)
                        s = re.sub(', which', ',this/these', s)
                        s = re.sub(r'\.$', '', s)

                        if s.count(',')==1:
                            ex_output_content_linked.append(s)
                        else:
                            ex_output_content.append(s)
                else:
                   pass

        print(len(ex_output_content_linked))
        ex_output_content_linked = list(set(ex_output_content_linked))
        for line in ex_output_content_linked:
            line = line.lower()
            if 'figure' not in line:
            #if line.count(',') <= 1:
                t_sline_ex = line.strip().split(',')
            #print("outpib"+str(t_sline_ex))
            for concept in t_sline_ex:
                #print("outpib" + str(concept))
                words = nltk.word_tokenize(concept)
                tagged = nltk.pos_tag(words)
                #print(tagged)
                parameters_list = []
                compteur = 0
                compt_Id_ex += 1
                tagged = nltk.pos_tag(word_tokenize(concept))
                tags = [word for word, pos in tagged if pos == 'VBZ' or pos == 'VBP'  or pos ==  'VBG' or pos == 'MD' or pos == 'JJR']
                if len(tags) < 1:
                    continue
                # classifyT = SentenceClassifier(concept)
                # polarite = classifyT.classifySentence()
                classifyT = ClassifyWithIncr_it()
                polarite = classifyT.main(concept)
                # if polarite == 'neutre':
                #     classify = SentenceClassifier(concept)
                #     polarite = classify.classifySentence()
                    # print(concept)

                get_parameters = ParameterExtractor(concept)
                parameters = get_parameters.extract_parameters()

                parameters_list.extend( parameters)
                # parameters_list=", ".join(parameters_list)
                # parameters_list = parameters_list
                #print("Index is: ")
                #print(t_sline_ex.index(concept))
                #print(concept)

                clean_concept = self.last_cleansing(concept)
                # if polarite == 'neutre':
                #     words = word_tokenize(clean_concept)
                #     hit = ' '.join([word + '/' + pos for word, pos in nltk.pos_tag(words)])
                #     noise_trash.append(hit)

                validity = self.discardLines(concept, 'referencing_indices')
                if t_sline_ex.index(concept) == 0 and validity is not None:
                    previous_polarity = polarite
                    values = OrderedDict({
                        "concept": {
                            "type": polarite,
                            "enfants": graphItemIdValue + str(compt_Id_ex + 1),
                            "id": graphItemIdValue + str(compt_Id_ex),
                            "sentence": clean_concept,
                            "source": output_file_name,
                            "parameters":parameters_list,
                            "image": urlImg,
                            "pdf": urlPDF
                        }

                    })

                else:
                    print("Previous polarity is : " + str(previous_polarity))
                    if previous_polarity =='partialSolution' or validity is None:
                        continue
                    else:
                        compteur += 1
                        values = OrderedDict({
                            "concept": {
                                "type": polarite,
                                "parents": graphItemIdValue + str(compt_Id_ex - 1),
                                "id": graphItemIdValue + str(compt_Id_ex),
                                "sentence": clean_concept,
                                "source": output_file_name,
                                "parameters": parameters_list,
                                "image": urlImg,
                                "pdf": urlPDF

                            }

                        })

                json_string_linkes = json.dumps(values, sort_keys=OrderedDict, indent=4, separators=(',', ': '))

                output_result.append(json_string_linkes)



        #for line in output_content:
                #print ("include=> "+line)
        #just examplification sentences
        #make a function of that
        ex_output_content = list(set(ex_output_content))
        for concept in ex_output_content:
            tagged = nltk.pos_tag(word_tokenize(concept))
            tags = [word for word, pos in tagged if
                    pos == 'VBZ' or pos == 'VBP' or pos == 'VBG' or pos == 'MD' or pos == 'JJR']
            if len(tags) < 1:
                continue
            parameters_list = []
            concept = concept.lower()
            compt_Id_ex += 1
            # classify = SentenceClassifier(sline)
            # polarite = classify.classifySentence()
            classifyT = ClassifyWithIncr_it()
            polarite = classifyT.main(concept)

            # if polarite =='neutre':
            #     classify = SentenceClassifier(concept)
            #     polarite = classify.classifySentence()
                # print(sline)

            #if polarite == 'partialSolution':
                #print(sline)
                #Insert a classifier here
            get_parameters = ParameterExtractor(concept)
            parameters = get_parameters.extract_parameters()

            clean_concept = self.last_cleansing(concept)
            parameters_list.extend(parameters)
            # if polarite == 'neutre':
            #     words = word_tokenize(clean_concept)
            #     hit = ' '.join([word + '/' + pos for word, pos in nltk.pos_tag(words)])
            #     noise_trash.append(hit)
            # parameters_list = ", ".join(parameters_list)
            validity = self.discardLines(concept, 'referencing_indices')
            if polarite != 'partialSolution' and validity is not None:

                values = OrderedDict({
                    "concept": {
                        "type": polarite,
                        "id": graphItemIdValue + str(compt_Id_ex),
                        "sentence": clean_concept,
                        "source": output_file_name,
                        "parameters": parameters_list,
                        "image": urlImg,
                        "pdf": urlPDF


                    }

                })
                json_string = json.dumps(values, sort_keys=OrderedDict, indent=4, separators=(',', ': '))
                output_result.append(json_string)



        for line in include_links:
            #print(line)
            #Put in lower case to improve matching
            line = line.lower()

            if re.match(r'however', line) and line.count(',') <= 1:
                line = str(line)
                sline = re.sub(r'however|,', '', line)
                if sline not in output_linked_content:
                    output_linked_content.append(sline)
            if re.match(r'however', line) and line.count(',') > 1:
                sline = re.sub(r'^however,?(\s\w+)\s*, that ', '', line)
                # sline = re.sub(r'however,.+, that ', '', sline)
                sline = re.sub(r'^however,?(\s\w+)+\s(above), ', '', sline)
                sline = re.sub(r'^however,?\s\w+ed(\s\w+)+,\s*', '', sline)
                sline = re.sub(r'^however,?\sif\s(desired|said)\s*,\s', '', sline)
                sline = re.sub(r'^however,?\s(it)\s(will be appreciated)\s*,\s(that)+\s*', '', sline)
                sline = re.sub(r'^however,?\s(as|if|because|when|since)\s*(?!is)', '', sline)
                sline = re.sub(r'^however,?\s*', '', sline)
                if sline not in output_linked_content:
                    output_linked_content.append(sline)
            if re.match(r'if', line) and line.count(',') <= 1:
                line = str(line)
                sline = re.sub(r'^if\s?(and when|not|desired|necessary)\s?,?\s*', '', line)
                sline = re.sub(r'^if,?\s*', '', sline)
                sline = re.sub(r'^if ', '', sline)
                if sline not in output_linked_content:
                    output_linked_content.append(sline)
                # print (sline)

            if re.match(r'when', line):
                line = str(line)
                line = line.lower()
                sline = re.sub(r'^when\s*', '', line)
                sline = re.sub(r'^when,?\s*', '', sline)
                sline = re.sub(r'^when ', '', sline)
                if sline not in output_linked_content:
                    output_linked_content.append(sline)
            if re.match(r'(^since)|(^\w+\s?,\s?since\s?)', line):
                sline = re.sub(r'^since', '', line)
                sline = re.sub(r'^\w+\s?,\s?since\s?', '', sline)
                if sline not in output_linked_content:
                    output_linked_content.append(sline)

        for line in output_content:
            line = line.lower()
            if re.match(r'if', line):
                line = str(line)
                sline = re.sub(r'^if ', '', line)
                if sline not in output_linked_content:
                    output_content.append(sline)
                #output_content.remove(line)

        uniq_output_linked_content = list(set(output_linked_content))
        for line in uniq_output_linked_content:
            #print("long sentences = > " + line)
            # line = str(i)
            #print(line)
            line = line.lower()
            if 'figure' in line:
                uniq_output_linked_content.remove(line)
            sline = re.sub(r'^\s+', '', line)
            sline = re.sub(r'^\d+\.+$', '', sline)

            if sline.count(',') <= 1:
                t_sline = tuple(sline.strip().split(', '))
                #print("outpib"+str(t_sline))
        for concept in t_sline:
            tagged = nltk.pos_tag(word_tokenize(concept))
            tags = [word for word, pos in tagged if
                    pos == 'VBZ' or pos == 'VBP' or pos == 'VBG' or pos == 'MD' or pos == 'JJR']
            if len(tags) < 1:
                continue
            else:
                parameters_list = []
                compteur = 0
                compt_Id += 1
                # classifyT = SentenceClassifier(concept)
                # polarite = classifyT.classifySentence()
                tagged = nltk.pos_tag(word_tokenize(concept))
                tags = [word for word, pos in tagged if pos.startswith('V') or pos == 'JJR']
                if len(tags) < 1:
                    continue
                classifyT = ClassifyWithIncr_it()
                polarite = classifyT.main(concept)


                # if polarite == 'neutre':
                #     classify = SentenceClassifier(concept)
                #     polarite = classify.classifySentence()
                    # print(concept)

                get_parameters = ParameterExtractor(concept)
                parameters = get_parameters.extract_parameters()

                parameters_list.extend( parameters)
                # parameters_list=", ".join(parameters_list)
                # parameters_list = parameters_list

                clean_concept = self.last_cleansing(concept)
                validity = self.discardLines(concept, 'referencing_indices')
                # if polarite == 'neutre':
                #     words = word_tokenize(clean_concept)
                #     hit = ' '.join([word + '/' + pos for word, pos in nltk.pos_tag(words)])
                #     noise_trash.append(hit)


                if t_sline.index(concept) == 0 and validity is not None:
                    previous_polarity = polarite
                    values = OrderedDict({
                        "concept": {
                            "type": polarite,
                            "enfants": graphItemIdValue + str(compt_Id + 1),
                            "id": graphItemIdValue + str(compt_Id),
                            "sentence": clean_concept,
                            "source": output_file_name,
                            "parameters":parameters_list,
                            "image": urlImg,
                            "pdf": urlPDF
                        }

                    })

                else:
                    print("Previous polarity is : " + str(previous_polarity))
                    if previous_polarity =='partialSolutiond' or validity is None:
                        continue
                    else:
                        compteur += 1
                        values = OrderedDict({
                            "concept": {
                                "type": polarite,
                                "parents": graphItemIdValue + str(compt_Id - 1),
                                "id": graphItemIdValue + str(compt_Id),
                                "sentence": clean_concept,
                                "source": output_file_name,
                                "parameters": parameters_list,
                                "image": urlImg,
                                "pdf": urlPDF

                            }

                        })

                json_string_linked = json.dumps(values, sort_keys=OrderedDict, indent=4, separators=(',', ': '))

                output_result.append(json_string_linked)


        uniq_output_content = list(set(output_content))
        for s in uniq_output_content:
            for y in uniq_output_content:
                if s != y:
                    result = Levenshtein.ratio(s, y)
                    if result > .7:
                        # print(s + " :IS SIMILAR TO: " + y)
                        if len(s) > len(y):
                            uniq_output_content.remove(y)
                        elif len(y) < len(s):
                            uniq_output_content.remove(s)


        for concept in uniq_output_content:
            tagged = nltk.pos_tag(word_tokenize(concept))
            tags = [word for word, pos in tagged if
                    pos == 'VBZ' or pos == 'VBP' or pos == 'VBG' or pos == 'MD' or pos == 'JJR']
            if len(tags) < 1:
                continue
            parameters_list = []
            concept = concept.lower()
            compt_Id += 1
            sline = re.sub(r'^if ', '', concept)
            sline = re.sub(r'^(if|preferably) ', '', sline)
            sline = re.sub(r'^\s+?said ', '', sline)
            # classify = SentenceClassifier(sline)
            # polarite = classify.classifySentence()
            classifyT = ClassifyWithIncr_it()
            polarite = classifyT.main(concept)
            # if polarite =='neutre':
            #     classify = SentenceClassifier(sline)
            #     polarite = classify.classifySentence()
                # print(sline)

            #if polarite == 'partialSolution':
                #print(sline)
                #Insert a classifier here
            get_parameters = ParameterExtractor(concept)
            parameters = get_parameters.extract_parameters()

            parameters_list.extend(parameters)
            # parameters_list = ", ".join(parameters_list)
            clean_concept = self.last_cleansing(sline)
            # if polarite == 'neutre':
            #     words = word_tokenize(clean_concept)
            #     hit = ' '.join([word + '/' + pos for word, pos in nltk.pos_tag(words)])
            #     noise_trash.append(hit)

            validity = self.discardLines(concept, 'referencing_indices')
            if polarite !='partialSolution' and validity is not None:

                values = OrderedDict({
                    "concept": {
                        "type": polarite,
                        "id": graphItemIdValue + str(compt_Id),
                        "sentence": clean_concept,
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