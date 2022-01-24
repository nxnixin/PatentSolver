# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:02:26 2016

@author: Achille Souili
"""
import re
import nltk



class ComplexParser(object):

    def __init__(self, sentence):
        self.sentence = sentence

    def extract_parameters(self):
        sentence = self.sentence
        concept = []


        words = nltk.word_tokenize(sentence)
        sentence = nltk.pos_tag(words)
        grammar = """CLAUSES: {<DT>?<JJ.*>?<DT><NN><.*>?<VB.*>?<.*>+}
                              """
        parameter_parser = nltk.RegexpParser(grammar)
        tree = parameter_parser.parse(sentence)
        for subtree in tree.subtrees():
            if subtree.label() == 'CLAUSES':
                #print(subtree)
                parameter_candidate = " ".join(word for word, tag in subtree.leaves())
                concept.append(parameter_candidate)
        concept = "d".join(concept)
        return concept

if __name__ == "__main__":

    Paragraph = "in which the surface of diffusion (24) is concave."
    words = nltk.word_tokenize(Paragraph)
    tagged = nltk.pos_tag(words)
    print(tagged)
    get_parameter = ComplexParser(Paragraph)
    parameters_list = get_parameter.extract_parameters()

    print (parameters_list)
