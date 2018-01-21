from commons.foldes_and_files import INPUT_FOLDER
#configurations for adjectives sense clustering

# MIN_ITEMS_FOR_CLUSTERING = 1000
MIN_ADJ_NOUN_OCCURRENCE = 10
MIN_UNIQUE_ITEMS_FOR_CLUSTERING = 50

ADJ_SUBSET_FOR_CLUSTERING_FILE_NAME = "v_10_50"#['hot','cold', 'warm', 'long', 'short', 'green']

MIN_NOUNS_WITHIN_PATTERN = 5 # Minimum number of times for 2 nouns co-occurrence to count as pattern pair
OUTLIERS_CLUSTERING_BY_PATTERNS = True #flag for clustering dbscan outliers using noun patterns instead of simple similarity check