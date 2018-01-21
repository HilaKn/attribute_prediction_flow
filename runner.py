import commons
from commons.foldes_and_files import *
import adj_sense_extractor.main
import we_trainer.main
import attribute_predictor.main
import argparse
# need this because when running the full flow, the AdjContext pickled adjectives are loaded
# and AdjContext should be under he main file
from adj_sense_extractor.main import AdjContext
import sys

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Attribute predictor')

    # adj_sense_extractor arguments
    parser.add_argument('sentences_input_file',help='input file path - sentences format')
    parser.add_argument('word_embeddings_file',help='word embeddings model file path')
    parser.add_argument('sentences_output_file',help='the generated file for WE training after adjectives clustering and labeling file path')
    parser.add_argument('-ss', '--only_sub_set',default=False,action='store_true', help='analyze only subset of adjectives from config file')
    parser.add_argument('-p', '--outliers_clustering_by_patterns',default=False,action='store_true', help='cluster dbscan outliers using patterns')

    # attribute_predictor arguments
    parser.add_argument('dev_file', help='dev input file')
    parser.add_argument('test_file', help='test input file')
    # parser.add_argument('we_file', help='word embeddings normed model file')
    # parser.add_argument('org_we_file', help='path to the original we model file - before adjectives clustering')
    parser.add_argument('-s', '--supervised', default=False, action='store_true',
                        help='train and evaluate also the supervised model')
    parser.add_argument('-oe', '--only_eval', default=False, action='store_true',
                        help='run only evaluation')
    args = parser.parse_args()





    we_input_file = os.path.join(OUTPUT_FOLDER, args.sentences_output_file)
    if not args.only_eval:
        # 1. Run the adjectives senses extractor
        sense_extractor_optional_indexes= []
        for opt_str in ['-ss', '-p']:
            try:
                idx = sys.argv.index(opt_str)
                sense_extractor_optional_indexes.append(idx)

            except ValueError:
                continue
        adj_sense_args = []
        adj_sense_args.extend(sys.argv[1:4])
        adj_sense_args.extend([sys.argv[idx] for idx in sense_extractor_optional_indexes])
        adj_sense_extractor.main.main(adj_sense_args)

        # 2. train word embedding on the prev script output
        we_trainer_args = [we_input_file]
        we_trainer.main.main(we_trainer_args)

    # 3. Evaluate the adjective senses
    attribute_predictor_optional_indexes= []
    try:
        idx = sys.argv.index('-s')
        attribute_predictor_optional_indexes.append(idx)
    except ValueError:
        pass

    we_senses_model_path = we_trainer.main.get_normed_we_full_path(we_input_file)
    attribute_predictor_args = [args.dev_file, args.test_file, we_senses_model_path, args.word_embeddings_file]
    attribute_predictor_args.extend([sys.argv[idx] for idx in attribute_predictor_optional_indexes])
    attribute_predictor.main.main(attribute_predictor_args)
