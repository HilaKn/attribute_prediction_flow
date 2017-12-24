from abc import ABCMeta, abstractproperty, abstractmethod
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from os import listdir
from os.path import isfile, join
import os
from commons.foldes_and_files import *
import logging
from config import *
from models import we_model

class Handlers:

    WORD_VECTOR = "word_vector_handler"
    PATTERNS = "patterns_handler"


class OutlierHandler(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.clustering_matrix = None
        self.clustering_input = None
        self.input_words = []
        self.shift_rows = 0

    def update_initial_data(self,clustering_matrix, clustering_input, input_words, shift_rows_by = 0):
        self.clustering_matrix = clustering_matrix
        self.clustering_input = clustering_input
        self.input_words = input_words
        self.shift_rows = shift_rows_by


    def get_label(self, outlier_idx):
        '''
        use this for clustering specified outlier index from the clustering_input
        :param outlier_idx:
        :return: the clustered label for the specified outlier index
        '''
        outlier_vec = self.get_outlier_vector(outlier_idx)
        cosine_sim_matrix = np.dot(outlier_vec, self.clustering_matrix.T)
        max_sim_row = np.argmax(cosine_sim_matrix) - self.shift_rows
        return max_sim_row


    def get_all_labels(self):
        '''
        Use this method for full clustering_input labels
        :return:labels list for the clustering input
        '''

        outliers_matrix = np.array(map(self.get_outlier_vector, [idx for idx, outlier in enumerate(self.input_words)])).squeeze()
        sim_matrix = np.dot(outliers_matrix, self.clustering_matrix.T)
        clustering = np.argmax(sim_matrix, axis=1)
        labels_list = clustering.tolist()
        return labels_list


    @abstractmethod
    def get_outlier_vector(self, outlier_idx):
        pass


class WordVecHandler(OutlierHandler):

    def get_outlier_vector(self, outlier_idx):
        return self.clustering_input[outlier_idx]


class PatternsHandler(OutlierHandler):

    def __init__(self):
        super(PatternsHandler, self).__init__()
        self.noun_to_nouns = self.__load_patterns_data()

    def __load_patterns_data(self):
        noun_to_nouns = defaultdict(lambda: defaultdict(int))
        pattern_files = [f for f in listdir(noun_patterns_path) if isfile(join(noun_patterns_path, f))]
        lemmatizer = WordNetLemmatizer()
        for file in pattern_files:
            file_path = os.path.join(noun_patterns_path, file)
            with open(file_path) as f:
                for row in f:
                    data = row.rstrip().split('\t')
                    try:
                        noun_1 = lemmatizer.lemmatize(data[0])
                        noun_2 = lemmatizer.lemmatize(data[1])
                    except :
                        logging.exception("Unexpected error. skipping nouns")
                        continue
                    count = int(data[4])
                    noun_to_nouns[noun_1][noun_2] += count
                    noun_to_nouns[noun_2][noun_1] += count

        final_dict = defaultdict(lambda: defaultdict(int))
        for noun, noun_dic in noun_to_nouns.iteritems():
            for context_noun, count in noun_dic.iteritems():
                if count > MIN_NOUNS_WITHIN_PATTERN:
                    final_dict[noun][context_noun] = count
        return final_dict

    def get_outlier_vector(self, outlier_idx):
        outlier = self.input_words[outlier_idx]

        nouns_to_count = self.noun_to_nouns[outlier] #list of tuples [(noun_1,count),(noun_2,count)..]
        noun_count_list = [(noun,count) for noun,count in nouns_to_count.iteritems() if noun in we_model.model.vocab]
        # noun_count_list = sorted(noun_count_list, key=lambda x: x[1], reverse=True)[:min(len(noun_count_list),4)]
        # outlier_count_in_pattern = 1
        # if noun_count_list:
        #     outlier_count_in_pattern = np.average([item[1] for item in noun_count_list])
        # noun_count_list.append((outlier, outlier_count_in_pattern))
        outlier_vec = self.clustering_input[outlier_idx]#np.zeros(300)
        if len(noun_count_list) >= 1:
            nouns = [item[0] for item in noun_count_list]
            nouns_matrix = np.array([we_model.word_vec(noun) for noun in nouns ]).squeeze()
            nouns_weights = [item[1] for item in noun_count_list]#weights are the co-occurrence counts
            if len(noun_count_list) == 1:
                outlier_vec = nouns_matrix
            else:
                outlier_vec = np.average(nouns_matrix, axis=0, weights=nouns_weights)

        return outlier_vec


def get_outlier_handler(handler):
    handler_class = None
    if handler == Handlers.WORD_VECTOR:
        handler_class = WordVecHandler()
    elif handler == Handlers.PATTERNS:
        handler_class = PatternsHandler()
    return handler_class



