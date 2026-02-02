import random
random.seed(42) # ERASE


# Hebrew diacritics
hebrew_diacritics = ['\u05B0', '\u05B4', '\u05B5', '\u05B6', '\u05B7', '\u05B8', '\u05B9', '\u05BB']
hataf = ['\u05B1', '\u05B2', '\u05B3']
kamatz_or_shva = ['\u05B8', '\u05B0']
shin_yemanit_and_smalit = ['\u05C1', '\u05C2']
holam_or_shuruk = ['\u05B9', '\u05BC']
dagesh = '\u05BC'
hirik = '\u05B4'
shva = '\u05B0'
tzeire = '\u05B5'


class heb_randomal_candidates_creator:
    
    """ Hebrew randomal candidates creator class """

    @staticmethod
    def corrupt_diacritization(word):
        """ Corupt the given diacritized word by changing one of the diacritics randomly """
        diacritics_indices = [(i, char) for i, char in enumerate(word) if char in hebrew_diacritics]
        if not diacritics_indices:
            new_diacritic = random.choice(hebrew_diacritics)
            new_word = word[:1] + new_diacritic
            if len(word) > 1:
                new_word += word[1:]
            return new_word
        index, old_diacritic = random.choice(diacritics_indices)
        new_diacritic = random.choice([d for d in hebrew_diacritics if d != old_diacritic])
        new_word = word[:index] + new_diacritic + word[index + 1:]
        return new_word


    def randomly_diacritize(self, word, original_diacritized_word):
        """ Diacritize the given word in a randomal way, but with some basic sense """
        diacritized_word = ""
        
        for i, char in enumerate(word):
            diacritized_word += char
            if char < 'א' or 'ת' < char: # catch all the 27 Hebrew letters
                continue
            is_last_char = (i == len(word) - 1)
            
            if char == 'ש':
                diacritic = random.choice(shin_yemanit_and_smalit)
                diacritized_word += diacritic
            if i == 0 and char in ['ע', 'ח', 'ה', 'א']:
                if random.random() < 0.5:
                    diacritic = random.choice(hebrew_diacritics)
                else:
                    diacritic = random.choice(hataf)
                diacritized_word += diacritic
            elif char == 'ו':
                if i == 0:
                    diacritized_word += shva
                else:
                    diacritic = random.choice(holam_or_shuruk)
                    diacritized_word += diacritic
            elif is_last_char:
                if char == 'ת' and random.random() < 0.7:
                    continue
                if char in ['ך', 'ת']:
                    diacritic = random.choice(kamatz_or_shva)
                    diacritized_word += diacritic
                    if random.random() < 0.3:
                        diacritized_word += dagesh
            elif word[i+1] == 'ו':
                continue
            elif word[i+1] == 'י':
                if (i+1 == len(word) - 1) or random.random() < 0.25:
                    diacritic = random.choice(hebrew_diacritics)
                    diacritized_word += diacritic
            elif i != 0 and char == 'י' and word[i-1] == diacritized_word[-2]:
                diacritic = hirik if random.random() < 0.75 else tzeire
                diacritized_word = diacritized_word[:-1] + diacritic
                if random.random() < 0.3:
                        diacritized_word += dagesh
                diacritized_word += char
            else:
                diacritic = random.choice(hebrew_diacritics)
                diacritized_word += diacritic
                if random.random() < 0.25 or \
                    i == 0 and char in ['ת', 'פ', 'כ', 'ד', 'ג', 'ב']:
                    diacritized_word += dagesh
        
        if diacritized_word == original_diacritized_word:
            diacritized_word = self.corrupt_diacritization(diacritized_word)

        return diacritized_word
    
    def get_candidates(self, raw_word, original_diacritized_word, num_candidates=1):
        """
        return list of randomal diacritization patterns for the given word, may return duplicates but the returned
        list will not contain the given original diacritization
        :raw_word: word to diacritize randomally
        :original_diacritized_word: the original diacritization of the word to diacritize, to prevent returning
            the original diacritization
        :num_candidates: number of randomal candidates to return
        """
        return [self.randomly_diacritize(raw_word, original_diacritized_word) for i in range(num_candidates)]


"""
using example:
    randomal_candidates_creator = heb_randomal_candidates_creator()
    randomal_candidates_creator.get_candidates("מסתגל", "מִסְתַּגֵּל", 2)
"""