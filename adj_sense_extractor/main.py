import sys
import argparse
from commons.foldes_and_files import *
from adj_sense_clustering import AdjSensesClusteringRunner
from models import we_model
import logging
from commons.logger import logger
from commons.parser_commons import *
import timeit
import adj_sense_clustering # for sharing logging purpose
from collections import defaultdict
import gzip

#like v26 with orgrnizing output for next steps and running for all adjectives and not just HeiPLAS
# multiple DBSCAN runnings + label outliers + separate cluster average for outliers
#single vector per noun (wihtout multiple counting)
#apply lemmatizer on the nouns (e.g. consider 'car' and 'cars' as the same noun)



# this class is placed here in order to load correctly the adjectives pickles - it has to be under the main file
# cause this is where it was originally placed
class AdjContext(object):

    def get_head_noun(self, dependency_row_data, full_sentence_data):
        # the head noun is not setting correctly sometimes by the parser.
        # so here is a simple heuristic to take the closest noun to the
        # adjective as the head noun in case the head noun is the root
        # or some non reasonable token (e.g. '.')
        head_id = int(dependency_row_data[DEP_ID_COLUMN])
        if head_id == 0 or full_sentence_data[head_id - 1][TOKEN_COLUMN] in NON_RELEVANT_HEAD:
            noun_candidates = [int(token[TOKEN_ID_COLUMN]) for token in full_sentence_data
                               if token[POS_COLUMN] in NOUN_TAGS]
            head_word = NO_HEAD_NOUN_TAG
            if len(noun_candidates) > 0:
                id = int(dependency_row_data[TOKEN_ID_COLUMN])
                closest_noun = (noun_candidates[0], abs(noun_candidates[0] - id))
                for cand in noun_candidates:
                    dist = abs(cand - id)
                    if dist < closest_noun[1]:
                        closest_noun = (cand, dist)
                head_id = closest_noun[0]
                head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]

        #If the dep relation is not to a noun:
        #1. If relation is conj go recursivly to search for the head noun
        #2. Else: set to unrecognized head (to be removed later
        elif full_sentence_data[head_id - 1][POS_COLUMN] not in NOUN_TAGS:
            head_word = NO_HEAD_NOUN_TAG
            if dependency_row_data[DEP_RELATION_COLUMN] == "conj":
                head_word = self.get_head_noun(full_sentence_data[head_id - 1],full_sentence_data)

        else:
             head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]

        return head_word

    def __init__(self,dependency_row_data, full_sentence_data,sentence_id):
        self.adj = dependency_row_data[TOKEN_COLUMN].lower()
        self.sentence_id = sentence_id
        self.token_id = int(dependency_row_data[TOKEN_ID_COLUMN])
        self.label_id = 0
        self.label = self.adj.lower()

        head_noun = self.get_head_noun(dependency_row_data, full_sentence_data)
        self.head_noun = head_noun.lower()

    def update_label(self,label_id):
        self.label_id = label_id
        self.label = "{}_<{}>".format(self.adj,label_id)


def main(args):
    # global start_time, parser, args, runner, stop
    start_time = timeit.default_timer()
    parser = argparse.ArgumentParser(description='Generate adjectives senses by nouns clustering')
    parser.add_argument('sentences_input_file', help='input file path - sentences format')
    parser.add_argument('word_embeddings_file', help='word embeddings model file path')
    # parser.add_argument('pickled_adj_folder',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',
                        help='the generated file for WE training after adjectives clustering and labeling file path')
    parser.add_argument('-ss', '--only_sub_set', default=False, action='store_true',
                        help='analyze only subset of adjectives from config file')
    parser.add_argument('-p', '--outliers_clustering_by_patterns', default=False, action='store_true',
                        help='cluster dbscan outliers using patterns')
    args = parser.parse_args(args)
    # logging.basicConfig(filename='adj_sense_extractor.log', level=logging.DEBUG)
    logger.info('start')
    logger.info("loading word embedding model from {}".format(args.word_embeddings_file))
    we_model.load_model(args.word_embeddings_file)
    runner = AdjSensesClusteringRunner(args.sentences_input_file, args.sentences_output_file, args.only_sub_set,
                                       args.outliers_clustering_by_patterns)
    runner.run()
    stop = timeit.default_timer()
    print "DONE!"
    print "Total running time {}".format(stop - start_time)


if __name__ == '__main__':

    main(sys.argv[1:])