from .abstract_candidates_creator import candidates_creator


class sorting_based_candidates_creator(candidates_creator):
    
    """ sorting-based candidates creator class """

    def __init__(
            self,
            candidates_dict,
            radius=1,
        ) -> None:
        self.candidates_dict = candidates_dict
        self.radius = radius
        self.length_to_sorted_words_arrays = self.initialize_length_to_sorted_words_arrays_dict(candidates_dict.keys())

    def initialize_length_to_sorted_words_arrays_dict(self, text):
        """
        return a dictionary that it's keys are words lengths, and the values are two
        lists of words. One sorted normally, and the second sorted from the last letters
        """
        length_to_sorted_words_arrays = dict()
        for word in text:
            length = len(word)
            if length not in length_to_sorted_words_arrays:
                length_to_sorted_words_arrays[length] = set()
            length_to_sorted_words_arrays[length].add(word)

        for key in length_to_sorted_words_arrays.keys():
            length_to_sorted_words_arrays[key] = [
                sorted(list(length_to_sorted_words_arrays[key])),
                sorted(list(length_to_sorted_words_arrays[key]), key=lambda x: x[::-1])
            ]
        return length_to_sorted_words_arrays

    @staticmethod
    def binary_search(lst, target, reverse=0):
        left, right = 0, len(lst) - 1
        while left <= right:
            mid = left + (right - left) // 2
            tmp = lst[mid][::-1] if reverse else lst[mid]
            if tmp == target:
                return mid
            elif tmp < target:
                left = mid + 1
            else:
                right = mid - 1
        return right
    
    def get_candidates(self, word):
        cands = set()
        if len(word) not in self.length_to_sorted_words_arrays:
            return cands
        for i in range(2):
            idx = self.binary_search(self.length_to_sorted_words_arrays[len(word)][i], word)
            if idx != -1:
                similars = self.length_to_sorted_words_arrays[len(word)][i][max(idx-self.radius, 0):idx+self.radius+1]
                # similars = [s for s in similars if s != word]
                sets = [self.candidates_dict[word] for word in similars if word in self.candidates_dict]
                similars = set()
                for s in sets:
                    similars = similars.union(s)
                [cands.add(element) for element in [self.replace_hebrew_letters(word, similar_word) for similar_word in similars]]
        return list(cands)
        # return cands

    def get_radius(self):
        return self.radius
    
    def set_radius(self, new_radius):
        self.radius = new_radius


"""
using example:
    cands = sorting_candidates_creator.get_candidates("היה")
    print(sorting_candidates_creator.get_radius())
    print(cands)
    print(len(cands))
    print(sorting_candidates_creator.set_radius(1))
    cands = sorting_candidates_creator.get_candidates("היה")
    print(sorting_candidates_creator.get_radius())
    print(cands)
    print(len(cands))
"""
