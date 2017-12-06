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
from save_adj_dict_v7 import AdjContext, ADJ_DIC_FOLDER
import save_adj_dict_v7 as s
import json
import jsonpickle
from itertools import groupby
import itertools
import io
import os
import sys
from nltk.stem import WordNetLemmatizer
from commons.foldes_and_files import *
from adj_sense_clustering import AdjSensesClusteringRunner
from we_model import we_model
#like v26 with orgrnizing output for next steps and running for all adjectives and not just HeiPLAS
# multiple DBSCAN runnings + label outliers + separate cluster average for outliers
#single vector per noun (wihtout multiple counting)
#apply lemmatizer on the nouns (e.g. consider 'car' and 'cars' as the same noun)

def generate_output_folders():
    folders = [OUTPUT_FOLDER, adj_clusters_path, adj_analysis_path]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adjectives senses by nouns clustering')

    parser.add_argument('sentences_input_file',help='input file path - sentences format')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    # parser.add_argument('pickled_adj_folder',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',help='the generated file for WE training after adjectives clustering and labeling file path')
    parser.add_argument('--only_sub_set','-ss',default=False,action='store_true', help='analyze only subset of adjectives from config file')
    args = parser.parse_args()


    we_model.load_model(args.word_embeddings_file)

    runner = AdjSensesClusteringRunner( args.sentences_input_file,args.sentences_output_file, args.only_sub_set)

    runner.run()

    print "DONE!"