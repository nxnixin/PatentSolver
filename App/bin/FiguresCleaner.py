# -*- coding: utf-8 -*-

import re
import nltk
import json

from App.bin import constants


class FiguresCleaner(object):

    def __init__(self, sections):
        self.sections = sections

    def clean_figures(self):
        sections = self.sections
        clean_content = []
        with open(constants.ASSETS + "wordAfterNumber", 'r') as l:
            after_words = l.read().splitlines()
            after_words_patterns = re.compile('|'.join(after_words))
        with open(constants.ASSETS + "wordBeforeNumber", 'r') as l:
            before_words = l.read().splitlines()
            before_words_patterns = re.compile('|'.join(before_words))

        #sections = sections.splitlines()
        words = nltk.word_tokenize(sections)
        tagged_words = nltk.pos_tag(words)
        for i in range(len(tagged_words)):
            if i < len(tagged_words) - 1:
                next_word = tagged_words[i + 1][0]
                current_word = tagged_words[i][0]
                previous_word = tagged_words[i - 1][0]
                currentWordTag = tagged_words[i][1]
                if currentWordTag == 'CD' and not re.match(after_words_patterns,
                                                           next_word) is not None and not re.match(
                    before_words_patterns, previous_word) is not None:
                    if re.search(r'\d', current_word) is not None:
                        continue
                else:
                    clean_content.append(current_word + " ")
            else:
                clean_content.append("\n")

        return clean_content