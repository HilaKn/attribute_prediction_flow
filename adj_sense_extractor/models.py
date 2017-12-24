import gensim
from nltk.stem import WordNetLemmatizer

class WeModel(object):
    def __init__(self):
        self.__model = None
        self.__is_initiate = False

    @property
    def model(self):
        if not self.__model:
            print "please load mode first"
            return
        else:
            return self.__model

    def load_model(self, path):
        print "Loading word vectors from {}".format(path)
        self.__model = gensim.models.KeyedVectors.load(path, mmap='r') .wv # mmap the large matrix as read-only
        self.__model.syn0norm = self.__model.syn0

    def word_vec(self, word):
        return self.__model.word_vec(word)

we_model = WeModel()

lemmatizer = WordNetLemmatizer()