from collections import namedtuple


class AdjNounAttribute:

    def __init__(self, adjective, noun, attribute):
        self.adj = adjective
        self.noun = noun
        self.attr = attribute

    def __str__(self):
        string = ' '.join([self.attr.upper(),self.adj, self.noun])
        return string

# AdjNounAttribute = namedtuple('AdjNounAttribute', 'adj noun attr')