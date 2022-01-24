#!/usr/bin/python3
# -*- coding: utf-8 -*
import sys
import os
import math
import xlsxwriter
from textblob import TextBlob as tb

class FindTechnologies(object):

    def __init__(self):

        print("Starting")

    def tf(word, blob):
        return (float)(blob.noun_phrases.count(word)) / (float)(len(blob.noun_phrases))


    def n_containing(word, bloblist):
        return sum(1 for blob in bloblist if word in blob.noun_phrases)


    def idf(word, bloblist):
        return math.log(len(bloblist) / (float)(1 + n_containing(word, bloblist)))


    def tfidf(word, blob, bloblist):
        return tf(word, blob) * idf(word, bloblist)


    # Create an excel file for validation purpose

    def get_technologies(self):
        folder_path = "C:/Users/asouili01/Documents/PatSemBeta-v3/Data/input/Gaggenau/"
        stopwords = open('C:/Users/asouili01/Documents/PIXSEB/Ressources/stopwords.txt', 'r').read().split('\r\n')
        bloblist = []

        filenamelist = []

        for path, dirs, files in os.walk(folder_path):
            for filename in files:
                print(filename)
                filenamelist.append(filename)
                name, extension = filename.split('.')
                filepath = folder_path + "/" + filename
                filehandler = open(filepath, "r",encoding="utf-8")

                content = filehandler.read()
                filteredtext = [t for t in content if t.lower() not in stopwords]
                filteredcontent = ''.join(filteredtext)
                blob = 'blob_' + name.lower()
                print (blob)
                blob = tb(filteredcontent.lower())
                bloblist.append(blob)

                print(bloblist)

        for i, blob in enumerate(bloblist):
            print("Top words in document {}".format(i + 1))
            scores = {word: tfidf(word, blob, bloblist) for word in blob.noun_phrases}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:5]:
                print("\tWord: {}, TF-IDF: {}".format(word, round(score, 10)))

