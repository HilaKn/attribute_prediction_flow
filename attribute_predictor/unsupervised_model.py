import os
from multi_sense_we_wrapper import MultiSenseWE
from commons.common_types import  AdjNounAttribute
import gensim
from torch.autograd import Variable
import torch
import numpy as np
import random
import operator
from scipy import spatial
import argparse
import torch.nn as nn
import torch.nn.functional as F
from commons.logger import logger
from config import *
from commons.foldes_and_files import *

class UnsupervisedModel(object):

    def __init__(self, data_handler):
        self.data = data_handler

    def run(self):
        self.test()

    def test(self):

        x_test = self.data.x_test
        y_test = self.data.y_test
        attr_vecs = self.data.attr_vecs

        logger.info("attr_vecs size = {}".format(len(attr_vecs)))
        logger.info("x test shape: " + str(x_test.shape))
        logger.info("y_test: " + str(y_test.shape))

        correct = 0.0
        correct_in_K = 0.0
        predictions = []
        unique_attributes = attr_vecs.keys()
        attr_vecs_ordered = np.array([self.data.we_wrapper.word_vec(attr) for attr in unique_attributes]).squeeze()
        for test in self.data.test:

            adj_label = self.data.we_wrapper.get_adj_name(test.adj,test.noun)
            adj_vec = self.data.we_wrapper.adj_vec_by_context(test.adj,test.noun)
            # adj_vec = we_wrapper.org_model.word_vec(test.adj)
            sim = np.dot(adj_vec,attr_vecs_ordered.T)
            all_attr_idx = sim.argsort()[-244:][::-1]
            attr_all_preds = [unique_attributes[i] for i in all_attr_idx]
            # attr_ids = sim.argsort()[-K:][::-1]
            # adj_preds = [unique_attributes[i] for i in attr_ids]
            adj_preds = attr_all_preds[:K]
            correct_pred_idx = attr_all_preds.index(test.attr)
            predictions.append((AdjNounAttribute(test.adj,test.noun,test.attr),adj_preds[0],adj_label, correct_pred_idx))
            if adj_preds[0] == test.attr:
                correct += 1
            if test.attr in adj_preds:
                correct_in_K += 1

        with open(unsupervised_results,'w') as file:
            for item in predictions:
                string = ' '.join([str(item[0]),item[1].upper(),item[2], str(item[3])])
                print >>file,string

        logger.info("----unsupervised results-----")
        logger.info("correct = {}, total: {}, accuracy: {}".format(correct, len(self.data.test), correct/len(self.data.test)))
        logger.info("correct_in_{} = {}, total: {}, accuracy: {}".format(K, correct_in_K, len(self.data.test), correct_in_K/len(self.data.test)))

