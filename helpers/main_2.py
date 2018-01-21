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
    adj_subset_file_name = "v3_{}_{}".format(MIN_ADJ_NOUN_OCCURRENCE, MIN_UNIQUE_ITEMS_FOR_CLUSTERING)
    adj_subset_full_path = os.path.join(adj_subset_folder_path, adj_subset_file_name)
    lemmatizer = WordNetLemmatizer()

    counter = 0
    with open(adj_subset_full_path, 'w') as out:

        for file_name in os.listdir(adj_pickles_path):
            counter += 1
            print "{}. processing [{}]".format(counter,file_name)
            full_path = os.path.join(adj_pickles_path, file_name)
            with open(full_path) as f:

                adj_contexts_json = json.load(f)
                adj_contexts = jsonpickle.decode(adj_contexts_json)

            if len(adj_contexts) < (MIN_UNIQUE_ITEMS_FOR_CLUSTERING*MIN_ADJ_NOUN_OCCURRENCE):
                continue
            for context in adj_contexts:
                context.head_noun = lemmatizer.lemmatize(context.head_noun)

            sorted_contexts_list = sorted(adj_contexts, key=lambda x: x.head_noun, reverse=True)
            grouped_contexts_list = [list(grouped_contexts) for head_noun, grouped_contexts in
                                     groupby(sorted_contexts_list, lambda x: x.head_noun)]
            unique_contexts = [contexts[0] for i, contexts in enumerate(grouped_contexts_list) if len(contexts) > MIN_ADJ_NOUN_OCCURRENCE]


            if len(unique_contexts) >= MIN_UNIQUE_ITEMS_FOR_CLUSTERING:
                adj = file_name +"\n"
                out.write(adj)

    print "DONE"



