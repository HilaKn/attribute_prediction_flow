import argparse
from gensim.models.wrappers import FastText
from gensim.models.word2vec import LineSentence, Word2Vec,KeyedVectors
import multiprocessing
import math
import time
import  os
import logging
from commons.foldes_and_files import we_model_folder
from config import *
from commons.logger import logger
import sys

def get_we_model_full_path(input_file):
    model_name = os.path.basename(input_file)
    model_output_file = "{}_{}_{}".format(model_name, DIMENSION, CONTEXT_WINDOW)
    output_file_path = os.path.join(we_model_folder, model_output_file)
    return output_file_path

def get_normed_we_full_path(input_file):
    model_full_path = get_we_model_full_path(input_file)
    normed_model_path = model_full_path +"_normed"
    return normed_model_path


def main(args):
    # global start_time, parser, args, free_cores, cores, uptime_data, load_avg, used_cores, sentences, model_name, model_path, normed_model_path, model, end_time, hours, rem, minutes, seconds
    start_time = time.time()
    # Set up command line parameters.
    parser = argparse.ArgumentParser(description='Train word2vec model.')
    parser.add_argument('input_file', help='input file path for the word embeddings training')
    args = parser.parse_args(args)
    free_cores = 1
    if PARALLEL_FLAG:
        cores = multiprocessing.cpu_count()
        uptime_data = os.popen("uptime").read().split()
        load_avg = float(uptime_data[-3].strip(','))  # take the load average of the last minute(the third from the end)
        used_cores = math.ceil(load_avg / cores)
        free_cores = min(cores - used_cores, MAX_CORES_TO_USE)
        logger.info("running with {} threads".format(free_cores))
    sentences = LineSentence(args.input_file)
    model_name = args.input_file
    model_path = get_we_model_full_path(model_name)
    normed_model_path = get_normed_we_full_path(model_name)
    logger.info("Start training word2vec on file: {}".format(args.input_file))
    model = Word2Vec(sentences, size=DIMENSION, alpha=LEARNING_RATE, window=CONTEXT_WINDOW,
                     workers=free_cores, iter=EPOCHS)
    logger.info("done word2ve training")
    logger.info("saving model to: {}".format(model_path))
    model.save(model_path)
    logger.info("saving normalized model to: {}".format(normed_model_path))
    model.init_sims(replace=True)
    model.save(normed_model_path)
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("total training time{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    main(sys.argv[1:])
