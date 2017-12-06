from parser_commons import *

class AdjContext:

    def get_head_noun(self, dependency_row_data, full_sentence_data):
        # the head noun is not setting correctly sometimes by the parser.
        # so here is a simple heuristic to take the closest noun to the
        # adjective as the head noun in case the head noun is the root
        # or some non reasonable token (e.g. '.')
        head_id = int(dependency_row_data[DEP_ID_COLUMN])
        if head_id == 0 or full_sentence_data[head_id - 1][TOKEN_COLUMN] in NON_RELEVANT_HEAD:
            noun_candidates = [int(token[TOKEN_ID_COLUMN]) for token in full_sentence_data
                               if token[POS_COLUMN] in NOUN_TAGS]
            head_word = NO_HEAD_NOUN_TAG
            if len(noun_candidates) > 0:
                id = int(dependency_row_data[TOKEN_ID_COLUMN])
                closest_noun = (noun_candidates[0], abs(noun_candidates[0] - id))
                for cand in noun_candidates:
                    dist = abs(cand - id)
                    if dist < closest_noun[1]:
                        closest_noun = (cand, dist)
                head_id = closest_noun[0]
                head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]

        #If the dep relation is not to a noun:
        #1. If relation is conj go recursivly to search for the head noun
        #2. Else: set to unrecognized head (to be removed later
        elif full_sentence_data[head_id - 1][POS_COLUMN] not in NOUN_TAGS:
            head_word = NO_HEAD_NOUN_TAG
            if dependency_row_data[DEP_RELATION_COLUMN] == "conj":
                head_word = self.get_head_noun(full_sentence_data[head_id - 1],full_sentence_data)

        else:
             head_word = full_sentence_data[head_id - 1][TOKEN_COLUMN]

        return head_word

    def __init__(self,dependency_row_data, full_sentence_data,sentence_id):
        self.adj = dependency_row_data[TOKEN_COLUMN].lower()
        self.sentence_id = sentence_id
        self.token_id = int(dependency_row_data[TOKEN_ID_COLUMN])
        self.label_id = 0
        self.label = self.adj.lower()

        head_noun = self.get_head_noun(dependency_row_data, full_sentence_data)
        self.head_noun = head_noun.lower()
        
    def update_label(self,label_id):
        self.label_id = label_id
        self.label = "{}_<{}>".format(self.adj,label_id)