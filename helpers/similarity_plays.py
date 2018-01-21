import gensim

we_file = "..output_all/we_models/wiki_adj_labeled_text_300_5_normed"
word = "low_<5>"

# load pre-trained, normalized word2ec
model = gensim.models.KeyedVectors.load(we_file, mmap='r').wv  # mmap the large matrix as read-only
model.syn0norm = model.syn0

most_sim = model.similar_by_word(word)
print most_sim
