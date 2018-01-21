import os

current_dir = os.path.dirname(__file__)

ADJ_ANALYSIS_FOLDER = "adj_analysis"
ADJ_CLUSTERS_FOLDER = "adj_clusters"
ADJ_PICKLES_FOLDER = "adj_dic_folder_pickle"
PERFORMANCE_FOLDER = os.path.join(current_dir,"../performance_analysis")
NOUN_PATTERNS_FOLDER = "noun_patterns"
ADJ_PATTERNS_FOLDER = "adj_patterns"
ADJ_SUBSET_FOLDER = "adj_subsets"

OUTPUT_FOLDER = os.path.join(current_dir,"../output_all")
INPUT_FOLDER = os.path.join(current_dir,"../../input_folder")

adj_clusters_path = os.path.join(OUTPUT_FOLDER, ADJ_CLUSTERS_FOLDER)
adj_analysis_path = os.path.join(OUTPUT_FOLDER, ADJ_ANALYSIS_FOLDER)
adj_pickles_path = os.path.join(INPUT_FOLDER, ADJ_PICKLES_FOLDER)
adj_pattern_path = os.path.join(INPUT_FOLDER, ADJ_PATTERNS_FOLDER, "adj_patterns_list")
noun_patterns_path = os.path.join(INPUT_FOLDER, NOUN_PATTERNS_FOLDER)
adj_subsets_folder = os.path.join(INPUT_FOLDER, ADJ_SUBSET_FOLDER)

clustering_performance_file = os.path.join(PERFORMANCE_FOLDER, "clustering")



WE_MODELS_FOLDER = "we_models"
we_model_folder = os.path.join(OUTPUT_FOLDER, WE_MODELS_FOLDER)

RESULTS_FOLDER = "results"
results_folder = os.path.join(OUTPUT_FOLDER, RESULTS_FOLDER)

correct_predictions_file = os.path.join(results_folder,"true_predictions")
false_prediction_file = os.path.join(results_folder,"false_predictions")
test_results = os.path.join(results_folder,"test_results")
unsupervised_results= os.path.join(results_folder, "unsupervised_results" )

adj_patterns_data_folder = os.path.join(OUTPUT_FOLDER, "adj_patterns")

all_output_folders = [adj_clusters_path,adj_analysis_path,we_model_folder,results_folder, PERFORMANCE_FOLDER, adj_patterns_data_folder]
for folder in all_output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)