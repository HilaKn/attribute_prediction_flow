# Generate adjectives couples based on occurrences in given patterns.
# The purpose is to examine the hypothesis that adjectives from the asme attribute group tends to appear
# within "X or Y" patterns (and similar) and attribute from different attributes are more likely to appear with "X and Y" kind of patterns
# The process:
# 1. Load list of patterns from file
# 2. Load large text with Part of Speech tags
# 3. Extract adjectives couples for each pattern
# 4. Output file per pattern with all the adjectives pairs found in this pattern
import argparse
from extract_symmetric_adj.utils import Utils
from commons.foldes_and_files import *
from commons.parser_commons import *
from extract_symmetric_adj.patterns_data import *

def run():

    utils = Utils()
    patterns = utils.load_to_list(adj_pattern_path)
    # and_pattern_analyzer = XandY_Analyzer(args.input_text_file)
    # and_pattern_analyzer.examine_the_and_pattern()

    patterns_data_handler = PatternDataHandlerOnSenses(patterns,args.senses_data_file,args.input_text_file)



    patterns_data_handler.extract_patterns_matching()
    patterns_data_handler.export_results(adj_patterns_data_folder)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adjectives with multi senses list.')

    # parser.add_argument('patterns_list_file',help='file containing list of all the symmetric patterns')
    parser.add_argument('input_text_file',help='large text file with POS tagging')
    parser.add_argument('senses_data_file', help='sentences file with adjectives senses if apply_on_senses_data is true')
    args = parser.parse_args()

    run()

    print "DONE"

