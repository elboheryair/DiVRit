from .abstract_candidates_creator import candidates_creator
import numpy as np
np.random.seed(42) #TODO: Erase
import faiss
import re


class KNN_candidates_creator(candidates_creator):
    
    """ KNN-based candidates creator class """

    def __init__(
            self,
            k,
            num_letters,
            candidates_dict_keys,
            candidates_dict,
            train_appearances_dict,
            eval_appearances_dict
    ) -> None:
        """
        :k: constant for the k nearest neighbours algorithm
        :num_letters: number of letters in the language script
        :candidates_dict: dict mapping raw words to their known diacritization
                          patterns from the corpus
        :candidates_dict_keys: raw words from the diacritized corpus
        :train_appearances_dict: map diacritized word from the training set to
                                 a list of it's appearances in the training set
        :eval_appearances_dict: map diacritized word from the evaluation set to
                                a list of it's appearances in the evaluation set
        """
        
        self.k = k
        self.num_letters = num_letters # 27 in Hebrew
        self.candidates_dict = candidates_dict
        self.candidates_dict_keys = candidates_dict_keys
        self.appearances_counter_dict = self.create_appearances_dict(train_appearances_dict, eval_appearances_dict)

        # mappers between letters to indices
        self.letter_to_idx = dict()
        self.idx_to_letter = dict()
        for i, c in enumerate(range(ord(u"\u0590")+64, ord(u"\u05EA")+1)): # todo generalize to many languages
            self.letter_to_idx[chr(c)] = i
            self.idx_to_letter[i] = chr(c)
        
        self.one_hot_vectors = dict()
        self.get_one_hot_vectors()
        self.indexes = dict()
        self.create_indexes()
        
    @staticmethod
    def create_appearances_dict(train_appearances_dict, eval_appearances_dict):
        appearances_dict = {}
        for k, v in train_appearances_dict.items():
            appearances_dict[k] = len(v)
        for k, v in eval_appearances_dict.items():
            if k not in appearances_dict:
                appearances_dict[k] = len(v)
            else:
                appearances_dict[k] = appearances_dict[k] + len(v)
        return appearances_dict
    
    @staticmethod
    def fix_shin_diacritics(word):
        """
        Ensures that any 'ש' in a Hebrew diacritized word has either a right shin (שׁ) or left shin (שׂ).
        If missing, it adds a right shin dot (שׁ).
        Also removes shin diacritics from letters that are not 'ש'.
        """
        RIGHT_SHIN = '\u05C1'  # ּשׁ (Right Shin)
        # LEFT_SHIN = '\u05C2'   # ּשׂ (Left Shin)

        # remove shin diacritics from any letter that isn't 'ש'
        word = re.sub(r'([^\u05E9])[\u05C1\u05C2]', r'\1', word)

        # add right shin diacritic if a 'ש' is missing a shin marker
        word = re.sub(r'(\u05E9)(?![\u05C1\u05C2])', r'\1' + RIGHT_SHIN, word)

        return word
    
    def word_to_one_hot(self, word):
        vec = np.zeros((len(word) * self.num_letters))
        for i, c in enumerate(word):
            factor = 1
            # if c in "ש":
            #     factor = 2.0
            if c in "אבהוחיכךפשת":
                factor = 1.5
            vec[i*self.num_letters + self.letter_to_idx[c]] += factor
        return vec

    def one_hot_to_word(self, vector):
        word = ""
        for idx in np.argwhere(vector!=0):
            word += self.idx_to_letter[idx[0] % self.num_letters]
        return word
    
    def get_one_hot_vectors(self):
        for key in self.candidates_dict_keys:
            if len(key) not in self.one_hot_vectors:
                self.one_hot_vectors[len(key)] = []
            self.one_hot_vectors[len(key)].append(self.word_to_one_hot(key))

    def create_indexes(self):
        for length, vectors in self.one_hot_vectors.items():
            index = None
            X = None

            X = np.array(vectors)
            # Build the index
            index = faiss.IndexFlatL2(self.num_letters * length)  # L2 distance (squared Euclidean)
            index.add(X)  # Add vectors to the index
            self.indexes[length] = (X, index)

    def get_candidates(self, word):
        n = len(word)
        if n not in self.indexes:
            return set()
        vec = self.word_to_one_hot(word).reshape(1, n * self.num_letters)

        # Search the nearest neighbors
        distances, indices = self.indexes[len(word)][1].search(vec, self.k)

        # Most similar vectors
        most_similar_vectors = self.indexes[len(word)][0][indices]

        cands = []
        m_cands = None
        similars = []
        for m in most_similar_vectors[0]:
            similar_word = self.one_hot_to_word(m)
            # print(similar_word)
            if similar_word == word and similar_word in self.candidates_dict:
                m_cands = [c for c in self.candidates_dict[similar_word]]
                continue

            if similar_word not in self.candidates_dict:
                continue
            # for c in self.candidates_dict[similar_word]:
            for c in sorted(self.candidates_dict[similar_word], key=lambda x: self.appearances_counter_dict[x], reverse=True):
                similars.append(c)
        if m_cands:
            tmp = similars
            similars = sorted(m_cands, key=lambda x: self.appearances_counter_dict[x], reverse=True)
            similars.extend(tmp)
        
        for element in [self.replace_hebrew_letters(word, source) for source in similars]:
            # cands.append(element)
            cands.append(self.fix_shin_diacritics(element))

        # # sort the candidates by diacritization patterns frequency
        # freq_counts = {}
        # for item in cands:
        #     freq_counts[item] = freq_counts.get(item, 0) + 1
        # sorted_cands = sorted(freq_counts.keys(), key=lambda x: -freq_counts[x])
        # return sorted_cands

        # keep only one candidate from each pattern, while maintaining the candidates order
        seen = set()
        unique_cands = []
        for item in cands:
            if item not in seen:
                seen.add(item)
                unique_cands.append(item)
        return unique_cands
    
    # def get_candidates(self, word):
    #     n = len(word)
    #     if n not in self.indexes:
    #         return set()
    #     vec = self.word_to_one_hot(word).reshape(1, n * self.num_letters)

    #     # Search the nearest neighbors
    #     distances, indices = self.indexes[len(word)][1].search(vec, self.k)

    #     # Most similar vectors
    #     most_similar_vectors = self.indexes[len(word)][0][indices]

    #     cands = []
    #     similars = []
    #     for m in most_similar_vectors[0]:
    #         similar_word = self.one_hot_to_word(m)

    #         if similar_word not in self.candidates_dict:
    #             continue
    #         for c in self.candidates_dict[similar_word]:
    #             similars.append(c)
        
    #     for element in [self.replace_hebrew_letters(word, source) for source in similars]:
    #         cands.append(element)

    #     # # sort the candidates by diacritization patterns frequency
    #     # freq_counts = {}
    #     # for item in cands:
    #     #     freq_counts[item] = freq_counts.get(item, 0) + 1
    #     # sorted_cands = sorted(freq_counts.keys(), key=lambda x: -freq_counts[x])
    #     # return sorted_cands

    #     # keep only one candidate from each pattern, while maintaining the candidates order
    #     seen = set()
    #     unique_cands = []
    #     for item in cands:
    #         if item not in seen:
    #             seen.add(item)
    #             unique_cands.append(item)
    #     return unique_cands
    
    # def get_candidates(self, word):
    #     n = len(word)
    #     if n not in self.indexes:
    #         return set()
    #     vec = self.word_to_one_hot(word).reshape(1, n * self.num_letters)

    #     # Search the nearest neighbors
    #     distances, indices = self.indexes[len(word)][1].search(vec, self.k)

    #     # Most similar vectors
    #     most_similar_vectors = self.indexes[len(word)][0][indices]

    #     cands = set()
    #     similars = set()
    #     for m in most_similar_vectors[0]:
    #         similar_word = self.one_hot_to_word(m)

    #         if similar_word not in self.candidates_dict:
    #             continue
    #         similars = similars.union(self.candidates_dict[similar_word])
    #         [cands.add(element) for element in [self.replace_hebrew_letters(word, source) for source in similars]]

    #     return cands

    def get_k(self):
        return self.k
    
    def set_k(self, k):
        self.k = k


"""
using example:
    k = 5
    num_letters = 27
    # candidates_dict_keys = 
    # candidates_dict = 
    knn_candidates_creator = KNN_candidates_creator(k, num_letters, candidates_dict_keys, candidates_dict)
    knn_candidates_creator.set_k(3)
    print(knn_candidates_creator.get_k())
    print(knn_candidates_creator.get_candidates("מסתגל"))
"""