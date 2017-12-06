import cPickle
import os
import argparse
import gensim
import gzip
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.externals import joblib
import math
from collections import defaultdict
import pickle
from commons.shared_objects import AdjContext
import json
import jsonpickle
from itertools import groupby
import itertools
import io
import sys
from nltk.stem import WordNetLemmatizer
from config import *
from commons.foldes_and_files import *
from we_model import we_model as model
import operator
#like v26 with orgrnizing output for next steps and running for all adjectives and not just HeiPLAS
# multiple DBSCAN runnings + label outliers + separate cluster average for outliers
#single vector per noun (wihtout multiple counting)
#apply lemmatizer on the nouns (e.g. consider 'car' and 'cars' as the same noun)

class AdjProcessor(object):

    def __init__(self, adj):
        self.adj = adj
        self.filtered_contexts_list = []
        self.unique_contexts = []
        self.data_mapper = {}
        self.clustering_input = None

    def prepare_data(self):
        with open(adj_pickles_path) as f:
            adj_contexts_json = json.load(f)
            adj_contexts = jsonpickle.decode(adj_contexts_json)

            for context in adj_contexts:
                context.head_noun = self.lemmatizer.lemmatize(context.head_noun)


            sorted_contexts_list = sorted(adj_contexts, key=lambda x: x.head_noun, reverse=True)
            grouped_contexts_list = [list(grouped_contexts) for head_noun, grouped_contexts in
                                     groupby(sorted_contexts_list, lambda x: x.head_noun)]


            index = 0
            for i, contexts in enumerate(grouped_contexts_list):
                if len(contexts) > MIN_ADJ_NOUN_OCCURRENCE:
                    for j in xrange(len(self.filtered_contexts_list), len(self.filtered_contexts_list) + len(contexts)):
                        self.data_mapper[j] = index
                    self.filtered_contexts_list.extend(contexts)
                    self.unique_contexts.append(contexts[0])
                    index += 1


            if len(self.unique_contexts) < MIN_UNIQUE_ITEMS_FOR_CLUSTERING:
                print "only [{}] unique contexts for [{}]. moving to next adj".format(len(self.unique_contexts), self.adj)
                self.filtered_contexts_list = []
                return

            print "Clustering adj: [{}] with [{}] contexts and [{}] unique contexts".format(self.adj,
                                                                                             len(self.filtered_contexts_list),
                                                                                             len(self.unique_contexts))

            self.clustering_input = np.array([model.word_vec(context.head_noun) for context in self.unique_contexts])
            print "input is ready. shape: {}".format(self.clustering_input.shape)


    def cluster(self):
        #TODO: fix this method (4.12.17)
        try:

            print "DBSCAN clustering [{}] for".format(self.adj)
            clustering_alg = DBSCAN(eps=0.4, min_samples=5, metric='cosine', algorithm='brute', n_jobs=20).\
                fit(self.clustering_input)
            k_1 = len(set(clustering_alg.labels_))
            print "done clustering [{}] with [{}] clusters".format(self.adj, k_1)
            outlier_idx = [idx for idx, label in enumerate(clustering_alg.labels_) if
                            label == -1]  #save all indexes of outlier samples
            outlier_input = np.array([self.clustering_input[i] for i in outlier_idx])
            print "DBSCAN clustering [{}] outliers".format(len(outlier_input))
            clustering_alg_2 = DBSCAN(eps=0.5, min_samples=5, metric='cosine', algorithm='brute', n_jobs=20).fit(
                outlier_input)
            k_2 = len(set(clustering_alg_2.labels_))
            # outlier_idx_2 = [idx for idx, label in enumerate(clustering_alg_2.labels_) if
            #                     label == -1]  #save all indexes of outlier samples
            # outlier_vecs_2 = [outlier_input[i] for i in outlier_idx_2]
            print "done clustering [{}] with [{}] clusters".format(self.adj, k_2)
            label_id_gap = k_1 - (1 if -1 in clustering_alg.labels_ else 0)

            label_to_contexts_vecs = defaultdict(list)
            for i, label in enumerate(clustering_alg.labels_):
                if label != -1:
                    context_vec = self.clustering_input[[i], :]
                    label_to_contexts_vecs[label].append(context_vec)


            for i, label in enumerate(clustering_alg_2.labels_):
                if label != -1:
                    context_vec = outlier_input[[i], :]
                    label_to_contexts_vecs[label + label_id_gap].append(context_vec)
                else:
                    label_to_contexts_vecs[-1].append(outlier_input[[i], :])


            print "Generate label to avg vector dictionary"
            label_to_matrix = {label: np.array(context_vecs).squeeze() for label, context_vecs in
                                label_to_contexts_vecs.iteritems()}
            label_to_avg = {label: np.average(matrix, axis=0) for label, matrix in label_to_matrix.iteritems()}

            sorted_labels = sorted(label_to_avg.items(), key=operator.itemgetter(0))
            sorted_labels_avg = np.array([item[1] for item in sorted_labels])
            print "Done generate label to avg vector dictionary"

            clustering_labels = clustering_alg.labels_
            for i, org_i in enumerate(outlier_idx):
                if clustering_alg_2.labels_[i] != -1:
                    clustering_labels[org_i] = clustering_alg_2.labels_[i] + label_id_gap
                else:
                    # print "try to find the best cluster for outlier"
                    cosine_sim_matrix = np.dot(self.clustering_input[org_i], sorted_labels_avg.T)
                    max_sim_row = np.argmax(cosine_sim_matrix) - (1 if -1 in clustering_alg_2.labels_ else 0)

                    clustering_labels[org_i] = max_sim_row

            print "before final labeling"
            self.final_labeling = [clustering_labels[self.data_mapper[i]] for i in xrange(0, len(self.filtered_contexts_list))]

            print "start file writing"
            k = len(label_to_avg.keys())
            if k > 2 or (k == 2 and -1 not in label_to_avg.keys()):
                self.output_clusters()#self.final_labeling
                print "done file writing"
            else:
                print "because no real clusters were found, [{}] want be written to file"
                self.final_labeling = None
            print "done clustering"
        except:
            print "Failed to cluster adjective: [{}]".format(self.adj)
            print sys.exc_info()
        finally:
            print "Finally"

    def output_clusters(self):
        label_to_contexts = defaultdict(list)
        for i, label in enumerate(self.final_labeling):
            head_noun = self.filtered_contexts_list[i].head_noun
            label_to_contexts[label].append(head_noun)


        label_to_unique_contexts = {label:set(contexts) for label,contexts in label_to_contexts.iteritems()}

        # print clusters for manual analysis
        k = len(label_to_unique_contexts)-(1 if label_to_unique_contexts.has_key(-1) else 0)
        output_path = os.path.join(adj_analysis_path, "{}_{}".format(self.adj,k))
        with open(output_path, 'w') as f:
            for label, contexts_words in label_to_unique_contexts.iteritems():
                f.write("label\t{}\n".format(label))
                f.write("\n".join([(str(label) + "\t" +word) for word in contexts_words]))
                f.write("\n")

        #print clusters average vector for later prediction
        # print "output clusters for later prediction"
        label_to_vec = defaultdict(np.array)
        for label,nouns in label_to_unique_contexts.iteritems():
            if label!=-1:
                contexts_array = np.array([model.word_vec(noun) for noun in nouns]).squeeze()
                print "context_array shape: {}".format(contexts_array.shape)
                label_to_vec[label] = np.average(contexts_array,axis=0)
                print "avg vec shape: {}".format(label_to_vec[label].shape)
        sorted_label_to_vec = sorted(label_to_vec.items(), key=operator.itemgetter(0))
        labels_matrix = np.array([item[1] for item in sorted_label_to_vec]).squeeze()
        print "labels matrix shape = {}".format(labels_matrix.shape)
        model_output_path = os.path.join(adj_clusters_path, self.adj)
        np.savetxt(model_output_path,labels_matrix)

    def run(self):
        self.prepare_data()
        if self.clustering_input:
            self.cluster()
            self.output_clusters()
        else:
            print "skipping adj {}".format(self.adj)


class AdjSensesClusteringRunner(object):

    def __init__(self, sentence_input_file, sentence_out_file,only_subset):
        self.input_file = sentence_input_file
        self.output_file = sentence_out_file
        self.analyze_subset_flag = only_subset
        self.lemmatizer = WordNetLemmatizer()
        self.__adj_list = []
        self.sent_to_labeled_adj = defaultdict(list)

    @property
    def adj_list(self):
        if not self.__adj_list:
            if self.analyze_subset_flag:
                self.__adj_list = ADJ_SUBSET_FOR_CLUSTERING
            else:
                self.__adj_list = [file for file in os.listdir(self.pickled_adj_folder)]
        return self.__adj_list


    def update_sentence_to_labeled_adj(self, adj_processor):
        print "before updating sent_to_labeled_adj"
        for i in xrange(0, len(adj_processor.filtered_contexts_list)):
            context = adj_processor.filtered_contexts_list[i]
            if adj_processor.final_labeling[i] != -1:  # update label only for the clustered
                context.update_label(adj_processor.final_labeling[i])
                self.sent_to_labeled_adj[context.sentence_id].append(context)
        print "after updating sent_to_labeled_adj"

    def update_text_corpus(self):
        # update corpus from original sentences file
        print "Start updating corpus with new adjectives labels"
        output_text_file = os.path.join(OUTPUT_FOLDER, self.output_file)
        with io.open(self.input_file, 'r', encoding='utf8') as fi, \
                io.open(output_text_file, 'w', encoding='utf8')as fo:

            sentence_id = 0

            for line in fi:
                if self.sent_to_labeled_adj.has_key(sentence_id):
                    line_data = line.split()
                    for context in self.sent_to_labeled_adj[sentence_id]:
                        line_data[context.token_id - 1] = context.label
                    output = ' '.join(line_data) + '\n'
                    fo.write(output)

                else:
                    fo.write(line)

                sentence_id += 1
                if (sentence_id % 100000 == 0):
                    print "update corpus:  sentence {}".format(sentence_id)
                    # break
        print "Done generating new sentences file"

    def run(self):

        adj_processors = [AdjProcessor(adj) for adj in self.adj_list]
        for adj_processor in adj_processors:
            adj_processor.run()
            if adj_processor.filtered_contexts_list:
                self.update_sentence_to_labeled_adj(adj_processor,)
            else:
                print "No update for sent_to_labeled_adj"

        print "Done clustering all adjectives"

        self.update_text_corpus()



