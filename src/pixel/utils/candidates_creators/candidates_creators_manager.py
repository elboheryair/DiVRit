import re
import random
random.seed(42) # ERASE
from .abstract_candidates_creator import candidates_creator
# from ..candidates import remove_diacritics


hebrew_diacritics_pattern = re.compile(r'[\u05B0-\u05BC\u05C1\u05C2\u05C7]')
non_hebrew_regex = re.compile(r'[^א-ת]')
non_hebrew_chars_and_diacritics_regex = re.compile(r'[^א-ת\u05B0-\u05BC\u05C1\u05C2\u05C7]')


def remove_diacritics(text):
    return hebrew_diacritics_pattern.sub('', text)


def remove_non_hebrew_chars(text):
    return re.sub(non_hebrew_regex, '', text)


def remove_non_hebrew_chars_and_diacritics(original_word):
    return re.sub(non_hebrew_chars_and_diacritics_regex, '', original_word)


class candidates_creators_manager:
    
    """ class of candidates creation manager that create a fixed-size set of candidates """

    def __init__(self, creators_list: list[candidates_creator], randomal_candidates_creator, num_candidates=2, training_mode=True):
        """
        :creators_list: list of algorithms to use, the candidates created according to the order of the creators in the list
        :randomal_candidates_creator: candidates creator that makes it possible to fill in the requested number of candidates
        """
        self.creators_list = creators_list
        self.randomal_candidates_creator = randomal_candidates_creator
        self.training_mode = training_mode
        self.num_candidates = num_candidates
        if self.training_mode:
            self.get_candidates_func = self.get_candidates_for_training_mode
        else:
            self.get_candidates_func = self.get_candidates_for_testing_mode

    @staticmethod
    def restore_diacritics(original_word, modified_hebrew_word):
        raw_original = remove_diacritics(original_word)
        restored_word = []
        idx = 0
        for char in raw_original:
            if '\u05d0' <= char <= '\u05ea': # Hebrew letters Unicode range
                restored_word.append(char)
                idx += 1
                while idx < len(modified_hebrew_word) and not '\u05d0' <= modified_hebrew_word[idx] <= '\u05ea':
                    restored_word.append(modified_hebrew_word[idx])
                    idx += 1
            else:
                restored_word.append(char)
        return ''.join(restored_word)
    
    def get_candidates_for_training_mode(self, word):
        cands = [word]
        raw_word = remove_non_hebrew_chars(word)
        
        for creator in self.creators_list:
            new_candidates = creator.get_candidates(raw_word)
            new_candidates = [nc for nc in new_candidates if nc not in cands]
            num_selections = min(self.num_candidates-len(cands), len(new_candidates))
            cands.extend(new_candidates[:num_selections])
            if len(cands) == self.num_candidates:
                break
        
        if len(cands) < self.num_candidates:
            new_cands = self.randomal_candidates_creator.get_candidates(raw_word, word, self.num_candidates - len(cands))
            cands.extend(new_cands)
        
        return cands
    
    def get_candidates_for_testing_mode(self, word):
        cands = []
        raw_word = remove_non_hebrew_chars(word)
        
        for creator in self.creators_list:
            new_candidates = creator.get_candidates(raw_word)
            new_candidates = [nc for nc in new_candidates if nc not in cands]
            num_selections = min(self.num_candidates-len(cands), len(new_candidates))
            cands.extend(new_candidates[:num_selections])
            if len(cands) == self.num_candidates:
                break
        
        if len(cands) < self.num_candidates:
            new_cands = self.randomal_candidates_creator.get_candidates(raw_word, word, self.num_candidates - len(cands))
            cands.extend(new_cands)
        
        return cands
        # cands = set()
        # raw_word = remove_non_hebrew_chars(word)
        
        # for creator in self.creators_list:
        #     new_candidates = creator.get_candidates(raw_word)
        #     new_candidates = new_candidates.difference(cands)
        #     num_selections = min(self.num_candidates-len(cands), len(new_candidates))
        #     cands = cands.union(random.sample(new_candidates, num_selections))
        #     if len(cands) == self.num_candidates:
        #         break
        
        # cands = list(cands)
        # if len(cands) == 0:
        #     new_cands = self.randomal_candidates_creator.get_candidates(raw_word, raw_word, self.num_candidates)
        #     cands.extend(new_cands)
        # return cands
    
    def get_candidates(self, word):
        # the candidates creation is based only on Hebrew letters
        only_heb_and_diacritics = remove_non_hebrew_chars_and_diacritics(word)
        candidates = self.get_candidates_func(only_heb_and_diacritics)
        # if there are no characters other than Hebrew or diacritics finish
        if word == only_heb_and_diacritics:
            return candidates
        # else add the rest of the characters in their place in the candidates
        else:
            original_candidates = []
            raw_word = remove_diacritics(word)
            for candidate in candidates:
                original_candidates.append(self.restore_diacritics(raw_word, candidate))
            return original_candidates
    
    def get_training_mode(self):
        return self.training_mode
    
    def set_training_mode(self, new_training_mode):
        if new_training_mode:
            self.training_mode = True
            self.get_candidates_func = self.get_candidates_for_training_mode
        else:
            self.training_mode = False
            self.get_candidates_func = self.get_candidates_for_testing_mode
    
    def get_num_candidates(self):
        return self.num_candidates

    def set_num_candidates(self, new_num_candidates):
        self.num_candidates = new_num_candidates

"""
using example:
    creators_list = [knn_candidates_creator, sorting_candidates_creator]
    # randomal_candidates_creator = 
    num_candidates = 2
    training_mode = True
    super_candidates_creator = candidates_creators_manager(creators_list, randomal_candidates_creator, num_candidates, training_mode=True)
    word = "מִסְתַּגֵּל"
    super_candidates_creator.set_num_candidates(5)
    super_candidates_creator.set_training_mode(True)
    cands = super_candidates_creator.get_candidates(word)
"""