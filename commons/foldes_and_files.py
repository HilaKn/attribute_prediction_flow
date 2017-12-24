import os

ADJ_ANALYSIS_FOLDER = "adj_analysis"
ADJ_CLUSTERS_FOLDER = "adj_clusters"
ADJ_PICKLES_FOLDER = "adj_dic_folder_pickle"
PERFORMANCE_FOLDER = "../performance_analysis"
NOUN_PATTERNS_FOLDER = "noun_patterns"
ADJ_PATTERNS_FOLDER = "adj_patterns"
OUTPUT_FOLDER = "../output_all"
INPUT_FOLDER = "../input_folder"

adj_clusters_path = os.path.join(OUTPUT_FOLDER, ADJ_CLUSTERS_FOLDER)
adj_analysis_path = os.path.join(OUTPUT_FOLDER, ADJ_ANALYSIS_FOLDER)
adj_pickles_path = os.path.join(INPUT_FOLDER, ADJ_PICKLES_FOLDER)
adj_pattern_path = os.path.join(INPUT_FOLDER, ADJ_PATTERNS_FOLDER)
noun_patterns_path = os.path.join(INPUT_FOLDER, NOUN_PATTERNS_FOLDER)

clustering_performance_file = os.path.join(PERFORMANCE_FOLDER, "clustering")

all_output_folders = [adj_clusters_path,adj_analysis_path ]
for folder in all_output_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)