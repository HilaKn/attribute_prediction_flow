from commons.logger import logger
import numpy as np
from commons.common_types import AdjNounAttribute

def read_HeiPLAS_data(file_path):
    with open(file_path) as f:
        input_list = [line.split() for line in f.readlines()]
    data = [AdjNounAttribute(item[1],item[2],item[0].lower()) for item in input_list]
    return data


class DataHandler(object):

    def __init__(self, we_wrapper):

        self.train = []
        self.test = []
        self.we_wrapper = we_wrapper
        self.attributes = ()
        self.attr_vecs = np.array
        self.x_train = np.array
        self.y_train = np.array
        self.x_test= np.array
        self.y_test= np.array


    def filter_data(self, triplets):
        logger.info("before filter missing words, samples: " + str(len(triplets)))
        filtered_data  = [samp for samp in triplets
                     if samp.adj in self.we_wrapper.vocab and samp.noun in self.we_wrapper.vocab
                     and samp.attr in self.we_wrapper.vocab ]
        logger.info("after filter missing words, samples: " + str(len(filtered_data)))
        x_matrix = np.array([self.we_wrapper.adj_vec_by_context(samp.adj,samp.noun) for samp in filtered_data])
        y_matrix = np.array([self.we_wrapper.word_vec(samp.attr) for samp in filtered_data])
        logger.info("x shape: "+ str(x_matrix.shape))
        logger.info("y_train: " + str(y_matrix.shape))

        return filtered_data, x_matrix, y_matrix

    def run(self, dev_triplets, test_triplets):
        logger.info("filter training samples")
        self.train, self.x_train, self.y_train = self.filter_data(dev_triplets)
        logger.info("filter test samples")
        self.test, self.x_test, self.y_test = self.filter_data(test_triplets)

        dev_attributes = set([triplet.attr for triplet in dev_triplets if triplet.attr in self.we_wrapper.vocab])
        test_attributes = set([triplet.attr for triplet in test_triplets if triplet.attr in self.we_wrapper.vocab])
        self.attributes = dev_attributes.union(test_attributes)

        self.attr_vecs  = {attr: self.we_wrapper.word_vec(attr) for attr in self.attributes}




