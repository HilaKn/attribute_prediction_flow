import json
from itertools import groupby
import io
import operator
import timeit

from sklearn.cluster import DBSCAN
import jsonpickle
from itertools import groupby
import itertools
import io
import sys
from nltk.stem import WordNetLemmatizer
from config import *
from commons.foldes_and_files import *
from models import we_model as model
from models import lemmatizer

from outliers_handler import *

RARE_CONTEXT = "rare"

#like v26 with orgrnizing output for next steps and running for all adjectives and not just HeiPLAS
# multiple DBSCAN runnings + label outliers + separate cluster average for outliers
#single vector per noun (wihtout multiple counting)
#apply lemmatizer on the nouns (e.g. consider 'car' and 'cars' as the same noun)

class AdjProcessor(object):

    def __init__(self, adj):
        self.adj = adj
        self.filtered_contexts_list = []
        self.unique_contexts = []
        self.data_mapper = {}
        self.clustering_input = np.array([])
        self.unclustered_contexts = [] # list of contexts bellow minimum occurrence threshold
        self.unclustered_labels = {}
        self.final_labeling = []

    def prepare_data(self):
        current_adj_pickle_path = os.path.join(adj_pickles_path,self.adj)
        with open(current_adj_pickle_path) as f:

            adj_contexts_json = json.load(f)
            adj_contexts = jsonpickle.decode(adj_contexts_json)

        for context in adj_contexts:
            context.head_noun = lemmatizer.lemmatize(context.head_noun)

        sorted_contexts_list = sorted(adj_contexts, key=lambda x: x.head_noun, reverse=True)
        grouped_contexts_list = [list(grouped_contexts) for head_noun, grouped_contexts in
                                 groupby(sorted_contexts_list, lambda x: x.head_noun)]

        index = 0
        for i, contexts in enumerate(grouped_contexts_list):
            if len(contexts) > MIN_ADJ_NOUN_OCCURRENCE:
                for j in xrange(len(self.filtered_contexts_list), len(self.filtered_contexts_list) + len(contexts)):
                    self.data_mapper[j] = index
                self.filtered_contexts_list.extend(contexts)
                self.unique_contexts.append(contexts[0])
                index += 1
            else:
                self.unclustered_contexts.extend(contexts)
        print len(self.filtered_contexts_list), len(self.unique_contexts)


        if len(self.unique_contexts) < MIN_UNIQUE_ITEMS_FOR_CLUSTERING:
            logger.info( "only [{}] unique contexts for [{}]. moving to next adj".format(len(self.unique_contexts), self.adj))
            self.filtered_contexts_list = []
            return

        logger.info("Clustering adj: [{}] with [{}] contexts and [{}] unique contexts".format(self.adj,
                                                                                         len(self.filtered_contexts_list),
                                                                                         len(self.unique_contexts)))

        self.clustering_input = np.array([model.word_vec(context.head_noun) for context in self.unique_contexts])
        logger.info("input is ready. shape: {}".format(self.clustering_input.shape))


    # def get_outlier_label(self, outlier_vec, final_outliers_flag, sorted_labels_avg):
    #     # print "try to find the best cluster for outlier"
    #
    #     cosine_sim_matrix = np.dot(outlier_vec, sorted_labels_avg.T)
    #     max_sim_row = np.argmax(cosine_sim_matrix) - final_outliers_flag
    #     return max_sim_row

    def cluster(self, outlier_handler):
        try:
            performance_file = open(clustering_performance_file, 'a')
            start = timeit.default_timer()
            logger.info("DBSCAN clustering [{}]".format(self.adj))
            clustering_alg = DBSCAN(eps=0.4, min_samples=5, metric='cosine', algorithm='brute', n_jobs=1).\
                fit(self.clustering_input)
            #todo
            end_dbscan_1 = timeit.default_timer()

            k_1 = len(set(clustering_alg.labels_))

            logger.info("done clustering [{}] with [{}] clusters".format(self.adj, k_1))

            outlier_idx = [idx for idx, label in enumerate(clustering_alg.labels_) if
                            label == -1]  #save all indexes of outlier samples
            outlier_input = np.array([self.clustering_input[i] for i in outlier_idx])


            logger.info("DBSCAN clustering [{}] outliers".format(len(outlier_input)))

            start_db_scan_2 = timeit.default_timer()
            clustering_alg_2 = DBSCAN(eps=0.5, min_samples=5, metric='cosine', algorithm='brute', n_jobs=1).fit(
                outlier_input)
            end_all_dbscan = timeit.default_timer()

            k_2 = len(set(clustering_alg_2.labels_))
            # outlier_idx_2 = [idx for idx, label in enumerate(clustering_alg_2.labels_) if
            #                     label == -1]  #save all indexes of outlier samples
            # outlier_vecs_2 = [outlier_input[i] for i in outlier_idx_2]
            logger.info("done clustering [{}] with [{}] clusters".format(self.adj, k_2))

            label_id_gap = k_1 - (1 if -1 in clustering_alg.labels_ else 0)

            label_to_contexts_vecs = defaultdict(list)
            for i, label in enumerate(clustering_alg.labels_):
                if label != -1:
                    context_vec = self.clustering_input[[i], :]
                    label_to_contexts_vecs[label].append(context_vec)


            for i, label in enumerate(clustering_alg_2.labels_):
                if label != -1:
                    context_vec = outlier_input[[i], :]
                    label_to_contexts_vecs[label + label_id_gap].append(context_vec)
                else:
                    label_to_contexts_vecs[-1].append(outlier_input[[i], :])


            logger.info("Generate label to avg vector dictionary")

            label_to_matrix = {label: np.array(context_vecs).squeeze() for label, context_vecs in
                                label_to_contexts_vecs.iteritems()}
            label_to_matrix.pop(-1, None)

            label_to_avg = {label: np.average(matrix, axis=0) for label, matrix in label_to_matrix.iteritems()}

            sorted_labels = sorted(label_to_avg.items(), key=operator.itemgetter(0))
            sorted_labels_avg = np.array([item[1] for item in sorted_labels])

            logger.info("Done generate label to avg vector dictionary")



            # outliers_from_1st_labeled_at_2nd = [for i, org_i in enumerate(outlier_idx)  if clustering_alg_2.labels_[i] != -1]
            # final_outliers_flag = 1 if -1 in clustering_alg_2.labels_ else 0
            outliers_words = [context.head_noun for context in self.unique_contexts]
            outlier_handler.update_initial_data(sorted_labels_avg
                                            , self.clustering_input, outliers_words)

            clustering_labels = clustering_alg.labels_
            for i, org_i in enumerate(outlier_idx):
                if clustering_alg_2.labels_[i] != -1:
                    clustering_labels[org_i] = clustering_alg_2.labels_[i] + label_id_gap
                else:
                    clustering_labels[org_i] = outlier_handler.get_label(org_i)
            ######TRY TO REPLACE THE ABOVE
            # only_first_time_outliers = [(i,org_i) for i, org_i in enumerate(outlier_idx) if clustering_alg_2.labels_[i] != -1]
            # for (i,org_i) in only_first_time_outliers:
            #     clustering_labels[org_i] = clustering_alg_2.labels_[i] + label_id_gap
            # final_outliers = [(i,org_i) for i, org_i in enumerate(outlier_idx) if clustering_alg_2.labels_[i] == -1]
            # outliers_labels = outlier_handler.get_all_labels([org_i for i,org_i in final_outliers])
            # for (i,org_i) in final_outliers:
            #     clustering_labels[org_i] = outliers_labels[i]
            #
            ######TRY TO REPLACE THE ABOVE

            logger.info("before final labeling")

            self.final_labeling = [clustering_labels[self.data_mapper[i]]
                                   for i in xrange(0, len(self.filtered_contexts_list))]

            logger.info("start file writing")

            k = len(label_to_avg.keys())
            if k > 2 or (k == 2 and -1 not in label_to_avg.keys()):
                self.output_clusters()#self.final_labeling
                logger.info("done file writing")
            else:
                logger.info("because no real clusters were found, [{}] won't be written to file".format(self.adj))
                self.final_labeling = None

            end = timeit.default_timer()
            performance_file.write("{}\t{}\t{}\t{}\t{}\n".format(self.clustering_input.shape[0],
                                                         end-start, #all cluster method
                                                         end_all_dbscan-start,#both dbscan inner processing
                                                         end_dbscan_1 - start, #dbscan 1
                                                         end_all_dbscan - start_db_scan_2)) #dbscan 2
            logger.info("done clustering")
        except:
            logger.exception("Failed to cluster adjective: [{}]".format(self.adj))
        finally:
            performance_file.close()

    def output_clusters(self):
        label_to_contexts = defaultdict(list)
        for i, label in enumerate(self.final_labeling):
            head_noun = self.filtered_contexts_list[i].head_noun
            label_to_contexts[label].append(head_noun)

        label_to_unique_contexts = {label:set(contexts) for label,contexts in label_to_contexts.iteritems()}

        # print clusters for manual analysis
        k = len(label_to_unique_contexts)-(1 if label_to_unique_contexts.has_key(-1) else 0)
        output_path = os.path.join(adj_analysis_path, "{}_{}".format(self.adj,k))
        with open(output_path, 'w') as f:
            for label, contexts_words in label_to_unique_contexts.iteritems():
                f.write("label\t{}\n".format(label))
                f.write("\n".join([(str(label) + "\t" +word) for word in contexts_words]))
                f.write("\n")

        # print clusters average vector for later prediction
        label_to_vec = defaultdict(np.array)
        for label,nouns in label_to_unique_contexts.iteritems():
            if label!=-1:
                contexts_array = np.array([model.word_vec(noun) for noun in nouns]).squeeze()
                # print "context_array shape: {}".format(contexts_array.shape)
                label_to_vec[label] = np.average(contexts_array,axis=0)
                # print "avg vec shape: {}".format(label_to_vec[label].shape)
        sorted_label_to_vec = sorted(label_to_vec.items(), key=operator.itemgetter(0))
        labels_matrix = np.array([item[1] for item in sorted_label_to_vec]).squeeze()
        logger.debug("labels matrix shape = {}".format(labels_matrix.shape))
        model_output_path = os.path.join(adj_clusters_path, self.adj)
        np.savetxt(model_output_path,labels_matrix)
        return labels_matrix

    def set_label_for_unclustered(self, outlier_handler, labels_matrix):
        #clustering_matrix, clustering_input, input_words, shift_rows_by = 1
        indexes_for_clustering = [(i,context.head_noun) for i,context in enumerate(self.unclustered_contexts) if context.head_noun in model.model.vocab]
        clustering_input = np.array([model.word_vec(context_tuple[1]) for context_tuple in indexes_for_clustering]).squeeze()
        input_words = [context_tuple[1] for context_tuple in indexes_for_clustering]
        outlier_handler.update_initial_data(labels_matrix, clustering_input, input_words)
        data_labels = outlier_handler.get_all_labels()

        all_data_labels = [RARE_CONTEXT]*len(self.unclustered_contexts)
        for i,label in enumerate(data_labels):
            org_index = indexes_for_clustering[i][0]
            all_data_labels[org_index] = label
        return all_data_labels

    def run(self, outlier_handler):
        self.prepare_data()
        if self.clustering_input.size:
            self.cluster(outlier_handler)
            if self.final_labeling:
                labels_matrix = self.output_clusters()
                self.unclustered_labels = self.set_label_for_unclustered(outlier_handler, labels_matrix)
        else:
            logger.info("skipping adj {}".format(self.adj))


class AdjSensesClusteringRunner(object):

    def __init__(self, sentence_input_file, sentence_out_file,only_subset, outliers_clustering_by_pattern_flag=True):
        self.input_file = sentence_input_file
        self.output_file = sentence_out_file
        self.analyze_subset_flag = only_subset
        self.__adj_list = []
        self.sent_to_labeled_adj = defaultdict(list)
        if outliers_clustering_by_pattern_flag:
            handler = Handlers.PATTERNS
        else:
            handler = Handlers.WORD_VECTOR
        self.outlier_handler = get_outlier_handler(handler, )

    @property
    def adj_list(self):
        if not self.__adj_list:
            if self.analyze_subset_flag:
                adj_subset_file = os.path.join(adj_subsets_folder, ADJ_SUBSET_FOR_CLUSTERING_FILE_NAME)
                with open(adj_subset_file) as f:
                    self.__adj_list = f.read().splitlines()
                 # self.__adj_list= ADJ_SUBSET_FOR_CLUSTERING_FILE_NAME
            else:
                self.__adj_list = [file for file in os.listdir(adj_pickles_path)]
        return self.__adj_list


    def update_sentence_to_labeled_adj(self, adj_processor):
        logger.info("before updating sent_to_labeled_adj")

        for i in xrange(0, len(adj_processor.filtered_contexts_list)):
            context = adj_processor.filtered_contexts_list[i]
            context.update_label(adj_processor.final_labeling[i])
            self.sent_to_labeled_adj[context.sentence_id].append(context)

        # updating from the filtered data - nouns below threshold occurrences
        for i in xrange(0, len(adj_processor.unclustered_contexts)):
            context = adj_processor.unclustered_contexts[i]
            context.update_label(adj_processor.unclustered_labels[i])
            self.sent_to_labeled_adj[context.sentence_id].append(context)

        logger.info("after updating sent_to_labeled_adj")

    def update_text_corpus(self):
        # update corpus from original sentences file
        logger.info("Start updating corpus with new adjectives labels")
        output_text_file = os.path.join(OUTPUT_FOLDER, self.output_file)
        with io.open(self.input_file, 'r', encoding='utf8') as fi, \
                io.open(output_text_file, 'w', encoding='utf8')as fo:

            sentence_id = 0

            for line in fi:
                if self.sent_to_labeled_adj.has_key(sentence_id):
                    line_data = line.split()
                    for context in self.sent_to_labeled_adj[sentence_id]:
                        line_data[context.token_id - 1] = context.label
                    output = ' '.join(line_data) + '\n'
                    fo.write(output)

                else:
                    fo.write(line)

                sentence_id += 1
                if (sentence_id % 100000 == 0):
                    logger.debug("update corpus:  sentence {}".format(sentence_id))
                    # break
        print "Done generating new sentences file"

    def run(self):

        adj_processors = [AdjProcessor(adj) for adj in self.adj_list]
        logger.info("Starting to cluster {} adjectives".format(len(adj_processors)))
        count = 1
        for adj_processor in adj_processors:
            logger.info("working on adjective number = [{}] from [{}]".format(count, len(adj_processors)))
            adj_processor.run(self.outlier_handler)
            if adj_processor.final_labeling:
                self.update_sentence_to_labeled_adj(adj_processor)
            else:
                logger.info("No update for sent_to_labeled_adj")
            count+=1

        print "Done clustering all adjectives"
        logger.info("Done clustering all adjectives")
        self.update_text_corpus()



