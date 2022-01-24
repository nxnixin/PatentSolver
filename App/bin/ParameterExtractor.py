# -*- coding: utf-8 -*-

import re
import nltk
import Levenshtein
from App.bin import constants

class ParameterExtractor(object):

    def __init__(self, sentence):
        self.sentence = sentence

    def clean_parameter(self, parameter):
        line = re.sub(r'\s[a-zA-Z]$', r'', parameter)
        line = line.strip()
        return line

    def extract_parameters(self):
        sentence = self.sentence
        parameters_list = []
        with open(constants.ASSETS + "parameter_core", 'r') as l:
            words_list = l.read().splitlines()
            match_word = re.compile(r'(\b(?:%s)\b)' % '|'.join(words_list))

        with open(constants.ASSETS + "exclude_from_parameters", 'r') as m:
            not_included_words_list = m.read().splitlines()
            match_not_included_word = re.compile(r'(\b(?:%s)\b)' % '|'.join(not_included_words_list))

        parameter_indice = re.search(match_word, sentence)
        if parameter_indice:
            words = nltk.word_tokenize(sentence)
            sentence = nltk.pos_tag(words)
            grammar = """PARAMETER:{<NN>+<IN><DT>?<NN.*>+}
                                {<NN*>+}
                        """
            parameter_parser = nltk.RegexpParser(grammar)
            tree = parameter_parser.parse(sentence)
            for subtree in tree.subtrees():
                if subtree.label() == 'PARAMETER':
                    parameter_candidate = " ".join(word for word, tag in subtree.leaves())
                    parameter_candidate_indice = re.search(match_word, parameter_candidate)
                    not_parameter = re.search(match_not_included_word, parameter_candidate)
                    if parameter_candidate_indice and not not_parameter :
                        #parameter_candidate=self.clean_parameter(parameter_candidate)
                        parameters_list.append(parameter_candidate)
        parameters_list = list(set(parameters_list))



        return list(parameters_list)

