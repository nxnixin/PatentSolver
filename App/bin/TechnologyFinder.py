#!/usr/bin/python3
# -*- coding: utf-8 -*
import sys
import os
import math
import re

from App.bin import constants

from textblob import TextBlob as tb

class TechnologyFinder(object):

    def __init__(self, corpus):
        self.corpus = corpus

        print("Extracting technologies")

    def last_cleansing(self, tech):
        tech = str(tech)
        tech = re.sub(r'\s?\bcomprises\b', '', tech)
        return tech

    def get_technologies(self):

        corpus = self.corpus

        technologies = []
        def tf(word, blob):
            return (float)(blob.noun_phrases.count(word)) / (float)(len(blob.noun_phrases))

        def n_containing(word, bloblist):
            return sum(1 for blob in bloblist if word in blob.noun_phrases)

        def idf(word, bloblist):
            return math.log(len(bloblist) / (float)(1 + n_containing(word, bloblist)))

        def tfidf(word, blob, bloblist):
            return tf(word, blob) * idf(word, bloblist)

        stopwords = open(constants.ASSETS+'stopwords', 'r').read().split('\r\n')
        bloblist = []
        filenamelist = []

        for filepath,patent in corpus.items():

            filename = os.path.basename(os.path.normpath(filepath))
            #name, extension = filename.split('.')
            filenamelist.append(filepath)

            filteredtext = [t for t in patent if t.lower() not in stopwords]
            filteredcontent = ''.join(filteredtext)
            blob = tb(filteredcontent.lower())
            bloblist.append(blob)

        for i, blob in enumerate(bloblist):
            filename = []
            technologies.append(filename)
            scores = {word: tfidf(word, blob, bloblist) for word in blob.noun_phrases}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:6]:
                word = self.last_cleansing(word)
                print("techologies found")
                filename.append(word)

        technologies_list = dict(zip(filenamelist, technologies))
        return technologies_list

