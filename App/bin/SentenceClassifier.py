# -*- coding: utf-8 -*-


import nltk
from App.bin import constants

class SentenceClassifier(object):
    def __init__(self, sentence):
        self.sentence = sentence
        print("Classification....")


    def classifySentence(self):

        sentence = self.sentence

        def bagOfWords(labelled):
            wordsList = []
            for (words, sentiment) in labelled:
                wordsList.extend(words)
            return wordsList

        def wordFeatures(wordList):
            wordList = nltk.FreqDist(wordList)
            wordFeatures = wordList.keys()
            return wordFeatures

        def extract_Features(doc):
            docWords = set(doc)
            feat = {}
            for word in wordFeatures:
                feat['contains(%s)' % word] = (word in docWords)
            return feat


        with open(constants.ASSETS+"trainingsNegative") as l:
            problems = [tuple(map(str, i.strip().split(':'))) for i in l]
        with open(constants.ASSETS+"trainingsPositive") as f:
            solutions = [tuple(map(str, i.strip().split(':'))) for i in f]

        labelled = []
        for (words, polarity) in solutions + problems:
            words_filtered = [e.lower() for e in nltk.word_tokenize(words) if len(e) >= 3]
            labelled.append((words_filtered, polarity))



        wordFeatures = wordFeatures(bagOfWords(labelled))

        training_set = nltk.classify.apply_features(extract_Features, labelled)

        classifier = nltk.NaiveBayesClassifier.train(training_set)

        #print(classifier.show_most_informative_features(32))


        #print (sentence)
        #print("{0} \n Polarity: {1} \n".format(sentence, classifier.classify(extract_Features(sentence.split()))))
        classes = classifier.classify(extract_Features(sentence.split()))
        return classes