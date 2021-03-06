from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SharpClassifier(object):
    def __init__(self, sentence):
        self.sentence = sentence
        print("Classification....")

    def classify(self):
        sentence = self.sentence
        n_instances = 100
        subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
        obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
        len(subj_docs), len(obj_docs)

        train_subj_docs = subj_docs[:80]
        test_subj_docs = subj_docs[80:100]
        train_obj_docs = obj_docs[:80]
        test_obj_docs = obj_docs[80:100]
        training_docs = train_subj_docs + train_obj_docs
        testing_docs = test_subj_docs + test_obj_docs

        sentim_analyzer = SentimentAnalyzer()
        all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

        unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)

        sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
        training_set = sentim_analyzer.apply_features(training_docs)
        test_set = sentim_analyzer.apply_features(testing_docs)

        trainer = NaiveBayesClassifier.train
        classifier = sentim_analyzer.train(trainer, training_set)
        # for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
        #     print('{0}: {1}'.format(key, value))

        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(sentence)
        polarity = ''
        if ss['neg'] < ss['pos']:
            polarity = 'partialSolution'
        elif ss['neg'] > ss['pos']:
            polarity = 'problem'
        else:
            polarity ='neutre'
        # for k in sorted(ss):
        #     print('{0}: {1}, '.format(k, ss[k]), end='')
        return polarity



