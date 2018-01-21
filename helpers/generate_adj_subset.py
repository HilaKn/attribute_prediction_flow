import os
import argparse
import jsonpickle
import json
from nltk.stem import WordNetLemmatizer
from itertools import groupby
from adj_sense_extractor.config import *
from adj_sense_extractor.main import *
from commons.foldes_and_files import *

if __name__ == '__main__':



    adj_subset_folder_path = os.path.join(INPUT_FOLDER, "adj_subsets")
    adj_subset_file_name = "v_{}_{}".format(MIN_ADJ_NOUN_OCCURRENCE, MIN_UNIQUE_ITEMS_FOR_CLUSTERING)
    adj_subset_full_path = os.path.join(adj_subset_folder_path, adj_subset_file_name)

    with open(adj_subset_full_path, 'w') as out:

        for file_name in os.listdir(adj_analysis_path):
            adj = file_name.split('_')[0] +"\n"
            out.write(adj)

    print "DONE"
