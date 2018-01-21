from commons.logger import logger
import argparse
import os
from multi_sense_we_wrapper import MultiSenseWE
from data_processor import *
from hartungs_supervised_model import SupervisedModel
from unsupervised_model import UnsupervisedModel
from commons.foldes_and_files import *
import sys

def main(args):
    # global parser, args, dev_triplets, test_triplets, we_wrapper, data_handler, model
    parser = argparse.ArgumentParser(description='Train word2vec model.')
    parser.add_argument('dev_file', help='dev input file')
    parser.add_argument('test_file', help='test input file')
    parser.add_argument('we_file', help='word embeddings normed model file')
    # parser.add_argument('output_folder', help='path to the output folder')
    parser.add_argument('org_we_file', help='path to the original we model file - before adjectives clustering')
    parser.add_argument('-s', '--supervised', default=False, action='store_true',
                        help='train and evaluate also the supervised model')
    args = parser.parse_args(args)
    dev_triplets = read_HeiPLAS_data(args.dev_file)
    test_triplets = read_HeiPLAS_data(args.test_file)
    # load pre-trained, normalized word2ec
    we_wrapper = MultiSenseWE(args.org_we_file, args.we_file)
    we_wrapper.set_model()
    data_handler = DataHandler(we_wrapper)
    data_handler.run(dev_triplets, test_triplets)
    if args.supervised:
        model = SupervisedModel(data_handler)
        model.run()
    model = UnsupervisedModel(data_handler)
    model.run()
    logger.info("Done!!!!!")


if __name__ == '__main__':

    main(sys.argv[1:])