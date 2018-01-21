import os
from multi_sense_we_wrapper import MultiSenseWE
from commons.common_types import AdjNounAttribute
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
import logging
from config import *
from commons.foldes_and_files import *
from commons.logger import logger
from data_processor import DataHandler
#
# correct_predictions_file = "true_predictions"
# false_prediction_file = "false_predictions"
# test_results = "test_results"
unsupervised_results = "unsupervised_results_pre_clustering_wiki"


class Model(nn.Module):
    def __init__(self, D_in, D_out):
        super(Model, self).__init__()
        self.linear_1 = nn.Linear(D_in,D_out,bias=False)
        weights = np.identity(D_out)
        self.linear_1.weight.data = torch.Tensor(weights)

    def forward(self, x):
        return self.linear_1(x)


class SupervisedModel(object):

    def __init__(self, data_handler):
        self.data = data_handler

        nn_model = Model(D_IN, D_OUT)
        self.nn_model = nn_model
        self.criterion = torch.nn.MSELoss(size_average=True)#Mean Square Error loss
        self.optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-4)

    def run(self):
        self.online_training()
        self.test()


    def online_training(self,epochs = EPHOCS ):
        running_loss = 0.0
        y_train = self.data.y_train
        x_train = self.data.x_train
        indices = range(y_train.shape[0])
        for epoch in range(epochs):
            logger.info("Epoch: {}".format(epoch))
            random.shuffle(indices)
            for i in indices:

                x = Variable(torch.Tensor(x_train[[i]]))
                y = Variable(torch.Tensor(y_train[[i]]), requires_grad=False)

                # pytorch doesn't support directly in training without batching so this is kind of a hack
                x.unsqueeze(0)
                y.unsqueeze(0)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.nn_model(x)

                # Compute and print loss
                loss = self.criterion(y_pred, y)
                # print(epoch, loss.data[0])

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print statistics
                # running_loss += loss.data[0]
                # if i % 100 == 99:  #
                #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                #     running_loss = 0.0
        logger.info("Done online training")


    def test(self):
        we_wrapper = self.data.we_wrapper
        weights = self.nn_model.linear_1.weight.data.numpy()

        x_test =  self.data.x_test
        y_test = self.data.y_test
        attr_vecs = self.data.attr_vecs

        print "attr_vecs size = {}".format(len(attr_vecs))
        print "x test shape: " + str(x_test.shape)
        print "y_test: " + str(y_test.shape)
        print "weights shape: {}".format(weights.shape)

        x_test_matrix = np.dot(weights, np.transpose(x_test))
        print "x_test matrix shape = {}".format(x_test_matrix.shape)

        # check P@1 and P@5 accuracy
        correct = 0.0
        top_5_correct = 0.0
        correct_pred =[]
        false_pred = []
        results = []
        for i in xrange(0, x_test_matrix.shape[1]):
            y_pred = x_test_matrix[:, [i]]

            #calculate cosine similarity for normalized vectors
            cosine_sims = {attr: np.dot(y_pred.T, attr_vecs[attr]) for attr in attr_vecs.keys()}
            sorted_sims = dict(sorted(cosine_sims.iteritems(), key=operator.itemgetter(1), reverse=True)[:K])
            most_sim_attr = max(sorted_sims, key=lambda i: sorted_sims[i])
            if most_sim_attr == self.data.test[i].attr:
                correct += 1
                correct_pred.append(self.data.test[i])
            else:
                false_pred.append((self.data.test[i],most_sim_attr))
            if self.data.test[i].attr in sorted_sims.keys():
                top_5_correct += 1
            results.append((self.data.test[i],most_sim_attr))
        logger.info("supervised results")
        logger.info("correct: {} from total: {}. Accuracy: {}".format(correct, y_test.shape[0], correct / y_test.shape[0]))
        logger.info("top 5 correct: {} from total: {}. Accuracy: {}".format(top_5_correct, y_test.shape[0],
                                                                      top_5_correct / y_test.shape[0]))

        with open(correct_predictions_file,'w') as file:
            for item in correct_pred:
                # output = ' '.join([str(item), item[1].upper()])
                print >>file,item

        with open(false_prediction_file,'w') as file:
            for item in false_pred:
                output = ' '.join([str(item[0]), item[1].upper()])
                print >>file,output

        with open(test_results,'w')as file:
            for item in results:
                # output =  ' '.join([item[1].upper(), item[0]].adj, item[0].noun)
                print >>file,str(item[0])

        # correct = 0.0
        # correct_in_K = 0.0
        # predictions = []
        # unique_attributes = attr_vecs.keys()
        # # attr_vecs_2 = np.array([we_wrapper.org_model.word_vec(attr) for attr in unique_attributes]).squeeze()
        # attr_vecs_2 = np.array([we_wrapper.word_vec(attr) for attr in unique_attributes]).squeeze()
        # for test in filtered_test_samp:
        #     # print "{} {}".format(test.adj, test.noun)
        #
        #     adj_label = we_wrapper.get_adj_name(test.adj,test.noun)
        #     adj_vec = we_wrapper.adj_vec_by_context(test.adj,test.noun)
        #     # adj_vec = we_wrapper.org_model.word_vec(test.adj)
        #     sim = np.dot(adj_vec,attr_vecs_2.T)
        #     attr_ids = sim.argsort()[-K:][::-1]
        #     adj_preds = [unique_attributes[i] for i in attr_ids]
        #
        #     predictions.append((AdjNounAttribute(test.adj,test.noun,test.attr),adj_preds[0],adj_label))
        #     if adj_preds[0] == test.attr:
        #         correct += 1
        #     if test.attr in adj_preds:
        #         correct_in_K += 1
        #
        # file = open(args.output_folder +'/' + unsupervised_results,'w')
        # for item in predictions:
        #     string = ' '.join([item[0].attr.upper(),item[0].adj, item[0].noun,item[1].upper(),item[2]])
        #     print >>file,string
        # print "----unsupervised results-----"
        # print "correct = {}, total: {}, accuracy: {}".format(correct, len(filtered_test_samp), correct/len(filtered_test_samp))
        # print "correct_in_{} = {}, total: {}, accuracy: {}".format(K, correct_in_K, len(filtered_test_samp), correct_in_K/len(filtered_test_samp))
        #
