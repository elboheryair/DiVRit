from abc import ABC, abstractmethod
import re


hebrew_regex = re.compile(r'[א-ת]')


class candidates_creator(ABC):

    """ candidates creator abstract class """
    
    @abstractmethod
    def get_candidates(self, word):
        pass

    @staticmethod
    def replace_hebrew_letters(source, target):
        # extract Hebrew letters from the source and target strings
        source_hebrew = hebrew_regex.findall(source)
        target_hebrew_indices = [m.start() for m in hebrew_regex.finditer(target)]
        # replace Hebrew letters in the target string with those from the source string
        target_list = list(target)
        for idx, source_char in zip(target_hebrew_indices, source_hebrew):
            target_list[idx] = source_char
        return ''.join(target_list)