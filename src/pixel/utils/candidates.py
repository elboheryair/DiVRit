import torch
import numpy as np
np.random.seed(42) #TODO: Erase
import re
import random
random.seed(42) # ERASE
from pixel import get_attention_mask
from PIL import Image

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

non_hebrew_letters_pattern = '[^' + ''.join([chr(1488+i) for i in range(27)]) + ']'
words_with_spaces_pattern = r'(\S+)(\s*)'
# hebrew_diacritics_pattern = re.compile(r'[\\u05B0-\\u05BC\\u05C1\\u05C2]')
hebrew_diacritics_pattern = re.compile(r'[\u05B0-\u05BC\u05C1\u05C2\u05C7]')
hebrew_pattern = r'[\u0590-\u05FF\uFB1D-\uFB4F]+'
hebrew_regex = re.compile(r'[א-ת]')


def remove_diacritics(text):
    # Substitute diacritics with an empty strings
    return hebrew_diacritics_pattern.sub('', text)


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


def randomly_diacritize(word, original_diacritized_word):
    """ Diacritize the given word in a randomal way, but with some basic sense """
    diacritized_word = ""
    
    for i, char in enumerate(word):
        diacritized_word += char
        if char < 'א' or 'ת' < char: # catch all 27 Hebrew letters
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
        diacritized_word = corrupt_diacritization(diacritized_word)

    return diacritized_word


def replace_hebrew_letters(source, target):
  # extract Hebrew letters from the source and target strings
  source_hebrew = hebrew_regex.findall(source)
  target_hebrew_indices = [m.start() for m in hebrew_regex.finditer(target)]
  # replace Hebrew letters in the target string with those from the source string
  target_list = list(target)
  for idx, source_char in zip(target_hebrew_indices, source_hebrew):
      target_list[idx] = source_char
  return ''.join(target_list)


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


def get_candidates_from_similar_words(
    source,
    length_to_sorted_words_arrays,
    candidates_dict,
    radius=3
):
    cands = set()
    if len(source) not in length_to_sorted_words_arrays:
        return cands
    for i in range(2):
        idx = binary_search(length_to_sorted_words_arrays[len(source)][i], source)
        if idx != -1:
            similars = length_to_sorted_words_arrays[len(source)][i][max(idx-radius, 0):idx+radius+1]
            # similars = [s for s in similars if s != source]
            sets = [candidates_dict[word] for word in similars if word in candidates_dict]
            similars = set()
            for s in sets:
                similars = similars.union(s)
            [cands.add(element) for element in [replace_hebrew_letters(source, word) for word in similars]]
    return cands


def get_candidate2(example, num_candidates):
    # # to preserve the exact sentence, we keep the exact seperators between words
    # words_with_separators = re.findall(words_with_spaces_pattern, example)
    # separators = [sep for word, sep in words_with_separators]

    # raw_words = remove_diacritics(example).split()
    # words = example.split()

    # find all the hebrew words spans in the raw and diacritized sentence
    hebrew_regex = re.compile(hebrew_pattern)
    raw_example = remove_diacritics(example)
    diacritized_spans = [(match.start(), match.end()) for match in hebrew_regex.finditer(example)]
    raw_spans = [(match.start(), match.end()) for match in hebrew_regex.finditer(raw_example)]

    # extract only the Hebrew words and diacritics
    words = [example[l:r] for l, r in diacritized_spans]
    raw_words = [raw_example[l:r] for (l, r) in raw_spans]

    # define the characters between the Hebrew spans as seperators
    separators = [raw_example[raw_spans[i][1]:raw_spans[i+1][0]] for i in range(len(raw_spans)-1)]
    separators.append(raw_example[raw_spans[-1][1]:])

    randomal_diacritizations = []
    for word, raw_word in zip(words, raw_words):
        # if all the letter are Hebrew letters
        if False not in ['\u0590' <= c <= '\u05FF' for c in raw_word]:
            options = []
            for i in range(num_candidates-1): # create num_candidates-1 wrong options
                options.append(randomly_diacritize(raw_word, word))
            randomal_diacritizations.append(options)
        else:
            randomal_diacritizations.append([raw_word for _ in range(num_candidates)])
    
    labels = torch.randint(low=0, high=num_candidates, size=(len(words),))
    # labels = np.random.randint(low=0, high=num_candidates, size=len(words))
    candiadates = [first_word + example[diacritized_spans[0][1]:diacritized_spans[1][0]] \
                   for first_word in randomal_diacritizations[0]]
    # put the correct option in the right place
    candiadates.insert(labels[0], words[0] + separators[0])
    for i in range(1, len(words)):
        candiadates[labels[i]] += words[i] + separators[i] # correct option
        for j in range(num_candidates-1):
            candiadates[(labels[i]+j+1) % num_candidates] += randomal_diacritizations[i][j] + separators[i]

    return labels, candiadates


def get_candidate(example, candidates_dict, sorted_arrays_dict, num_candidates):
    # to preserve the exact sentence, we keep the exact seperators between words
    words_with_separators = re.findall(words_with_spaces_pattern, example)
    separators = [sep for word, sep in words_with_separators]

    raw_words = remove_diacritics(example).split()
    words = example.split()

    # # find all the hebrew words spans in the raw and diacritized sentence
    # hebrew_regex = re.compile(hebrew_pattern)
    # raw_example = remove_diacritics(example)
    # diacritized_spans = [(match.start(), match.end()) for match in hebrew_regex.finditer(example)]
    # raw_spans = [(match.start(), match.end()) for match in hebrew_regex.finditer(raw_example)]

    # # extract only the Hebrew words and diacritics
    # words = [example[l:r] for l, r in diacritized_spans]
    # raw_words = [raw_example[l:r] for (l, r) in raw_spans]

    # # define the characters between the Hebrew spans as seperators
    # separators = [raw_example[raw_spans[i][1]:raw_spans[i+1][0]] for i in range(len(raw_spans)-1)]
    # separators.append(raw_example[raw_spans[-1][1]:])

    wrong_diacritizations = []
    for word, raw_word in zip(words, raw_words):
        options = set()
        num_selections = 0
        
        # first, choose patterns of this word that appear in the corpus
        if raw_word in candidates_dict:
            num_selections = min(num_candidates-1, len(candidates_dict[raw_word]))
            options = set(random.sample(candidates_dict[raw_word], num_selections))
            options = options.difference(set(word)) # only wrong patterns
        
        # second (if needed), add patterns of similar words
        similar_candidates = get_candidates_from_similar_words(
            source=raw_word,
            length_to_sorted_words_arrays=sorted_arrays_dict,
            candidates_dict=candidates_dict,
            radius=1
        )
        similar_candidates = similar_candidates.difference(options).difference(set(word))
        num_selections = min(num_candidates-1-num_selections, len(similar_candidates))
        options = options.union(random.sample(similar_candidates, num_selections))
        
        # third (if needed), add random patterns
        options = list(options)
        for _ in range(num_candidates-1-len(options)): # total num_candidates-1 wrong options
            options.append(randomly_diacritize(raw_word, word))
        wrong_diacritizations.append(options)
    
    correct_options = torch.randint(low=0, high=num_candidates, size=(530,))
    candiadates = [first_word + separators[0] for first_word in wrong_diacritizations[0]]
    # put the correct option in the right place
    candiadates.insert(correct_options[0], words[0] + separators[0])
    for i in range(1, len(words)):
        candiadates[correct_options[i]] += words[i] + separators[i] # correct option
        for j in range(num_candidates-1):
            candiadates[(correct_options[i]+j+1) % num_candidates] += wrong_diacritizations[i][j] + separators[i]

    return correct_options, candiadates


def get_candidates(examples, candidates_dict, sorted_arrays_dict, num_candidates=2):
    """
    Manage the candidates creation. For each example, the function suggests
    num_candidates options to diacritize each.
    The candidates consist of the same words in the given examples, but with some
    changes in the diacritics. For each word, only one candidate will have the correct
    diacritization, and the others will have wrong diacritizations.
    The i'th labels list relates to the i'th list of candidates, that were created from
    the i'th example. The j'th label (in the i'th labels list) is the index of the
    candidate with the correct diacritization of the j'th word.
    
    :examples: diacritized sentences to create the candidates from
    :num_candidates: number of diacritization candidates to create for each word
    """
    labels_candidates = [
        get_candidate(example, candidates_dict, sorted_arrays_dict, num_candidates) for example in examples
    ]
    labels = [tup[0] for tup in labels_candidates]
    candidates = [tup[1] for tup in labels_candidates]
    return labels, candidates


def find_words_in_image2(sentence, pixel_values, eos):
    """
    Assuming there are k wordds in the given sentence, the function returns
    a (k,2) shaped matrix. 
    The i'th row holds two indices - (i1, i2). The first patch that the i'th
    word from the sentence appear at is the i1 patch, and the last patch indicated
    by (i2-1).
    """
    col_sum = pixel_values.sum(axis=0)
    
    # find words ending columns
    cs_end = col_sum[:eos] + col_sum[1:eos+1] + col_sum[2:eos+2] + col_sum[3:eos+3]
    # to catch space in the end
    cs_end[-3:] += np.array([4080, 8160, 12240], dtype='uint64')
    cond1 = (cs_end[:eos] == 16320)
    cond2 = (np.concatenate([np.array([16320]), cs_end[:eos-1]]) != 16320)
    ends = [np.where(cond1 & cond2)[0] - 1][0]
    
    # find words beginning columns
    cs_start = col_sum[:eos] + col_sum[1:eos+1] + col_sum[2:eos+2] + col_sum[3:eos+3]
    # to catch space in the beginning
    cs_start = np.concatenate([np.array([12240, 12240, 12240], dtype='uint64') + col_sum[:3], cs_start[:-3]])
    cond1 = (cs_start[:eos] == 16320)
    cond2 = (np.concatenate([cs_start[1:eos], np.array([16320])]) != 16320)
    starts = [np.where(cond1 & cond2)[0] + 1][0]
    
    # no offset, means that no start was found to the first word
    if col_sum[0] != 4080:
        starts = np.concatenate([np.array([0]), starts])
    
    # means that no end was found to the last word
    if col_sum[eos-1] != 4080:
        ends = np.concatenate([ends, np.array([eos-1])])

    # convert columns into patch indices
    ends = (ends // 16) + 1
    starts //= 16

    spans = np.concatenate([starts, ends]).reshape(2, starts.shape[0]).T
    return spans


def find_words_in_image(sentence, pixel_values, eos):
    """
    Assuming there are k wordds in the given sentence, the function returns
    a (k,2) shaped matrix. 
    The i'th row holds two indices - (i1, i2). The first patch that the i'th
    word from the sentence appear at is the i1 patch, and the last patch indicated
    by (i2-1).
    """
    col_sum = pixel_values.sum(axis=0)
    
    # find words ending columns
    cs_end = col_sum[:eos] + col_sum[1:eos+1] + col_sum[2:eos+2]
    # to catch space in the end
    cs_end[-2:] += np.array([4080, 8160], dtype='uint64')
    cond1 = (cs_end[:eos] == 12240)
    cond2 = (np.concatenate([np.array([12240]), cs_end[:eos-1]]) != 12240)
    ends = [np.where(cond1 & cond2)[0] - 1][0]
    
    # find words beginning columns
    cs_start = col_sum[:eos] + col_sum[1:eos+1] + col_sum[2:eos+2]
    # to catch space in the beginning
    cs_start = np.concatenate([np.array([8160, 8160], dtype='uint64') + col_sum[:2], cs_start[:-2]])
    cond1 = (cs_start[:eos] == 12240)
    cond2 = (np.concatenate([cs_start[1:eos], np.array([12240])]) != 12240)
    starts = [np.where(cond1 & cond2)[0] + 1][0]
    
    # no offset, means that no start was found to the first word
    if col_sum[0] != 4080:
        starts = np.concatenate([np.array([0]), starts])
    
    # means that no end was found to the last word
    if col_sum[eos-1] != 4080:
        ends = np.concatenate([ends, np.array([eos-1])])

    # convert columns into patch indices
    ends = (ends // 16) + 1
    starts //= 16

    spans = np.concatenate([starts, ends]).reshape(2, starts.shape[0]).T
    return spans


def get_words_spans(sentence, pixel_values, eos, renderer):
    """
    Manage the process of finding the spans of the words. Span is pair of
    indices, indicating the first and last (not including) patches in which a
    word is shown.
    If there are more spaces in the image than words in the sentence, it means
    that there are words with pairs of letters that the renderer draws with at
    least four blank columns between them (e.g. "11" and "||"). In this case, the
    function finds those words and fix their spans.
    """
    spans = find_words_in_image(sentence, pixel_values, eos)
    splits = sentence.split()
    if len(splits) != spans.shape[0]:
        for i, word in enumerate(splits):
            # if the word has only hebrew letters the problem doesn't arise
            if not re.search(non_hebrew_letters_pattern, word, flags=0):
                continue
            encoding = renderer(word)
            eos = encoding.num_text_patches * 16
            word_spans = find_words_in_image(word, encoding.pixel_values, eos)
            if word_spans.shape[0] > 1:
                spans[i][1] = spans[i + word_spans.shape[0] - 1][1]
                spans = np.concatenate([spans[:i+1], spans[i + word_spans.shape[0]:]])
    return spans


def get_attention_mask_for_specific_words(word_spans, to_mask, num_text_patches):
    """
    Get the spans of patches that the words appear at, and create an attention
    mask for some of the words.
    If to_mask is a number in [0,1], the number indicates the percentage of words
    to mask. If to_mask is a list, the words that will be masked are the words in
    the indices given in this list.
    """
    attention_mask = torch.ones(num_text_patches)
    if type(to_mask) is not list:
        size = int(word_spans.shape[0] * to_mask)
        to_mask = torch.torch.randperm(size)[:word_spans.shape[0]]
    for idx in to_mask:
        begin, end = word_spans[idx]
        attention_mask[begin:end] = 0
    return attention_mask


def get_length_to_sorted_words_arrays_dict(text):
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


class examples_preprocessor:
    def __init__(
            self,
            renderer,
            words_sampler,
            candidates_creator,
            transforms,
            rtl=True
        ):
        self.renderer = renderer
        self.words_sampler = words_sampler
        self.candidates_creator = candidates_creator
        self.transforms = transforms
        self.rtl = rtl
        self.prev_examples = None

    @staticmethod
    def flip_pixel_values(encoding):
        """ mirror the text image (pixel values), for right to left scripts """
        
        # find the most right blank column
        eos = encoding.sep_patches[0]
        cols_sum = np.sum(encoding.pixel_values[:,:eos], axis=0)
        right = eos - 1
        while right >= 0 and cols_sum[right] == 4080:
            right -= 1
        if eos == 528*16: # image width bigger than 8464
            left =  0
            while left < eos and cols_sum[left] == 4080:
                left += 1
        else:
            left = 2
        
        # flip only the text, and set offset of two columns
        end = right - left + 3 # for the offset and to include last column
        encoding.pixel_values[:, 2:end] = \
            np.flip(encoding.pixel_values[:, left:right+1], axis=1)
        encoding.pixel_values[:, 0:2] = 255
        encoding.pixel_values[:, end:eos] = 255
        
        return encoding
    
    def apply_renderer(self, text):
        encoding = self.renderer(text)
        # # encoding = self.renderer("    ".join(text.split())) # todo spacious
        if self.rtl:
            encoding = self.flip_pixel_values(encoding)
        return encoding
    
    @staticmethod
    # def compute_shift(offset):
    def compute_shift(offset, cand_num_patches, original_span):
        """
        return how many columns to move the text in the image, to make the candidate
        start in the same column that the raw word from the context image is start
        """
        # the text starts in the second patch
        script_shift = offset[0] - 2 + 16
        # # keep the two first columns blank
        # if script_shift < 2:
        #     script_shift += 16
        # set the number of patches the same as the number of original text patches
        eos_shift = 1 # the first patch is always blank
        original_num_patches = original_span[1] - original_span[0]
        if original_num_patches > cand_num_patches:
            eos_shift += 1
        return eos_shift, script_shift
    
    # def render_candidate(self, candidate, offset):
    def render_candidate(self, candidate, offset, span):
        """ render candidate into image and fix the offset to fit the raw image """
        encoding = self.apply_renderer(candidate)
        # to_shift_eos, cols_to_shift_script = self.compute_shift(offset)
        # # shift the eos first if needed, to ensure the eos is not overrided
        # eos = encoding.sep_patches[0]
        eos_shift, cols_to_shift_script = \
            self.compute_shift(offset, encoding.num_text_patches, span)
        # shift the eos first, to ensure the eos is not overrided
        prev_eos = encoding.sep_patches[0]
        cur_eos = prev_eos + eos_shift * 16
        encoding.sep_patches[0] = cur_eos
        encoding.num_text_patches += eos_shift
        encoding.pixel_values[:, prev_eos:cur_eos+16] = \
            np.roll(encoding.pixel_values[:, prev_eos:cur_eos+16], -16)
        encoding.pixel_values[:, :cur_eos] = \
            np.roll(encoding.pixel_values[:, :cur_eos], cols_to_shift_script)
        return encoding
    
    @staticmethod
    def find_spans_and_offsets_in_image(encoding):
        """
        Assuming there are k words in the given encoded sentence, the function
        returns two (k,2) shaped matrices. 
        The i'th row in the first matrix holds two indices - (i1, i2). The first
        patch that the i'th word from the sentence appear at is the i1 patch, and the
        last patch indicated by (i2-1).
        The i'th row in the second matrix holds the index of the word's start and the
        index of the word's end in the word's start and end patches.
        """
        
        # unpacking
        pixel_values = encoding.pixel_values
        eos = encoding.sep_patches[0]
        
        # spaces are at least 3 white columns, that are add up to 3*16*255 = 12240
        col_sum = pixel_values.sum(axis=0)
        
        # find words ending columns
        cs_end = col_sum[:eos] + col_sum[1:eos+1] + col_sum[2:eos+2]
        # to catch space in the end
        cs_end[-2:] += np.array([4080, 8160], dtype='uint64')
        cond1 = (cs_end[:eos] == 12240)
        cond2 = (np.concatenate([np.array([12240]), cs_end[:eos-1]]) != 12240)
        ends = [np.where(cond1 & cond2)[0] - 1][0]
        
        # find words beginning columns
        cs_start = col_sum[:eos] + col_sum[1:eos+1] + col_sum[2:eos+2]
        # to catch space in the beginning
        cs_start = np.concatenate([np.array([8160, 8160], dtype='uint64') + col_sum[:2], cs_start[:-2]])
        cond1 = (cs_start[:eos] == 12240)
        cond2 = (np.concatenate([cs_start[1:eos], np.array([12240])]) != 12240)
        starts = [np.where(cond1 & cond2)[0] + 1][0]
        
        # no offset, means that no start was found to the first word
        if col_sum[0] != 4080:
            starts = np.concatenate([np.array([0]), starts])
        
        # means that no end was found to the last word
        if col_sum[eos-1] != 4080:
            ends = np.concatenate([ends, np.array([eos-1])])

        # get the columns of the offsets
        end_offsets = ends % 16
        start_offsets = starts % 16
        offsets = np.concatenate([start_offsets, end_offsets]).reshape(2, start_offsets.shape[0]).T
        
        # convert columns into patch indices
        ends = (ends // 16) + 1
        starts //= 16
        spans = np.concatenate([starts, ends]).reshape(2, starts.shape[0]).T

        return spans, offsets
    
    def get_words_spans_and_offsets(self, sentence, encoding):
        """
        Manage the process of finding the spans of the words. Span is pair of
        indices, indicating the first and last (not including) patches in which a
        word is shown.
        If there are more spaces in the image than words in the sentence, it means
        that there are words with pairs of letters that the renderer draws with at
        least four blank columns between them (e.g. "11" and "||"). In this case, the
        function finds those words and fix their spans.
        Note: the spans are plus one, for the CLS embedding.
        """
        spans, offsets = self.find_spans_and_offsets_in_image(encoding)
        # validate that the spans and the offsets are aligned with the image
        splits = sentence.split()
        if len(splits) != spans.shape[0]:
            for i, word in enumerate(splits):
                # if the word has only hebrew letters the problem doesn't arise
                if not re.search(non_hebrew_letters_pattern, word, flags=0):
                    continue
                word_encoding = self.apply_renderer(word)
                word_spans, word_offsets = self.find_spans_and_offsets_in_image(word_encoding)
                if word_spans.shape[0] > 1: # fix the span and offset
                    spans[i][1] = spans[i + word_spans.shape[0] - 1][1]
                    spans = np.concatenate([spans[:i+1], spans[i + word_spans.shape[0]:]])
                    offsets[i][1] = offsets[i + word_offsets.shape[0] - 1][1]
                    offsets = np.concatenate([offsets[:i+1], offsets[i + word_offsets.shape[0]:]])
        # first embedding is CLS, so let's add 1 to every span
        spans += 1
        return spans, offsets
    
    def _preprocess_examples(self, examples):
        """Preprocess a batch of sentences and applying transforms."""
        
        # render raw examples into images
        raw_text = [remove_diacritics(example) for example in examples["text"]]
        raw_encodings = [self.apply_renderer(example) for example in raw_text]
        
        # for each example, sample words to train on
        indicess, wordss = self.words_sampler.sample_words(examples["text"])

        # get the patch spans that the words are found in, and the columns
        spanss = []
        offsetss = []
        for i, indices in enumerate(indicess):
            spans, offsets = self.get_words_spans_and_offsets(
                raw_text[i], raw_encodings[i]
            )
            # if len(spans) <= indices[-1]: # todo delete, only for debugging purposes
            #     print(examples["text"][i])
            spanss.append([spans[idx] for idx in indices])
            offsetss.append([offsets[idx] for idx in indices])
        
        # create labels
        labelss = torch.randint(
            low=0,
            high=self.candidates_creator.num_candidates,
            size=(len(examples["text"]), self.words_sampler.sampling_func.num_words)
        )
        
        # for each sampled word, suggest some diacritizations and render them
        candidatess_lists = []
        candidatess_encodings = []
        for i, words in enumerate(wordss): # (sentences, num_words, num_candidates)
            candidatess_lists.append([])
            candidatess_encodings.append([])
            for j, word in enumerate(words):
                candidates = self.candidates_creator.get_candidates(word)
                
                # raw_word = remove_diacritics(word) # todo check!!!
                # # candidates[-1] = raw_word # todo check!!!
                # candidates[-2], candidates[-1] = raw_word, raw_word # todo check!!!
                
                # shuffle the wrong candidates
                candidates[1:] = random.sample(candidates[1:], len(candidates)-1)
                
                # put the correct candidates where the labels pointing on by rotation
                candidates = candidates[-labelss[i][j]:] + candidates[:-labelss[i][j]]
                candidatess_lists[i].append(candidates)
                candidates_encodings = [
                    self.render_candidate(cand, offsetss[i][j], spanss[i][j]) for cand in candidates
                ]
                candidatess_encodings[i].append(candidates_encodings)

        # insert the preprocessed data to the batch dictionary
        examples['raw_text'] = raw_text
        examples['indicess'] = indicess
        examples['wordss'] = wordss
        examples['spanss'] = spanss
        examples['labelss'] = labelss
        examples['candidatess_lists'] = candidatess_lists
        examples['raw_examples'] = \
            [self.transforms(Image.fromarray(example.pixel_values)) \
                for example in raw_encodings]
        examples['raw_num_patches'] = \
            [example.num_text_patches for example in raw_encodings]
        examples['raw_attention_mask'] = \
            [get_attention_mask(n) for n in examples['raw_num_patches']]
        examples['candidates_examples'] = [
            [[self.transforms(Image.fromarray(ex.pixel_values)) for ex in candidates] \
                for candidates in candidatess] \
                    for candidatess in candidatess_encodings
        ]
        examples['candidatess_num_patches'] = [
            [candidates[0].num_text_patches for candidates in candidatess] \
                for candidatess in candidatess_encodings
        ]
        examples['candidates_attention_mask'] = [
            [get_attention_mask(n) for n in num_patches] \
                for num_patches in examples['candidatess_num_patches']
        ]
        return examples
    
    def preprocess_examples(self, examples):
        try:
            preprocessed_examples = self._preprocess_examples(examples)
            self.prev_examples = preprocessed_examples
        except:
            pass
        return self.prev_examples

class MM_examples_preprocessor(examples_preprocessor):
    def __init__(
            self, renderer, words_sampler, candidates_creator, transforms, tokenizer, rtl=True
        ):
        super().__init__(renderer, words_sampler, candidates_creator, transforms, rtl)
        
        self.tokenizer = tokenizer
        
    def custom_preprocessing(self, text):
        processed_text = ""
        for char in text:
            if char.isspace():
                processed_text += char
            elif len(self.tokenizer.tokenize(char)) == 0:
                processed_text += f" {self.tokenizer.unk_token} "
            else:
                processed_text += char
        return processed_text
    
    def _preprocess_examples(self, examples):
        """Preprocess a batch of sentences and applying transforms."""
        
        # render raw examples into images
        raw_text = [remove_diacritics(example) for example in examples["text"]]
        # replace unknown chars to [UNK], for keeping same number of words in text and raw_text
        raw_text = [self.custom_preprocessing(rt) for rt in raw_text]
        
        # for each example, sample words to train on
        indicess, wordss = self.words_sampler.sample_words(examples["text"])

        # create labels
        labelss = torch.randint(
            low=0,
            high=self.candidates_creator.num_candidates,
            size=(len(examples["text"]), self.words_sampler.sampling_func.num_words)
        )
        
        # for each sampled word, suggest some diacritizations and render them
        candidatess_lists = []
        candidatess_encodings = []
        for i, words in enumerate(wordss): # (sentences, num_words, num_candidates)
            candidatess_lists.append([])
            candidatess_encodings.append([])
            for j, word in enumerate(words):
                candidates = self.candidates_creator.get_candidates(word)
                
                # # # todo check - MM_examples_preprocessor
                # raw_word = remove_diacritics(word)
                # # candidates[-1] = raw_word
                # candidates[-2], candidates[-1] = raw_word, raw_word
                
                # shuffle the wrong candidates
                candidates[1:] = random.sample(candidates[1:], len(candidates)-1)
                
                # put the correct candidates where the labels pointing on by rotation
                candidates = candidates[-labelss[i][j]:] + candidates[:-labelss[i][j]]
                candidatess_lists[i].append(candidates)
                candidates_encodings = [
                    self.renderer(cand) for cand in candidates
                ]
                candidatess_encodings[i].append(candidates_encodings)

        # insert the preprocessed data to the batch dictionary
        examples['raw_text'] = raw_text
        examples['indicess'] = indicess
        examples['wordss'] = wordss
        examples['labelss'] = labelss
        examples['candidatess_lists'] = candidatess_lists
        examples['candidates_examples'] = [
            [[self.transforms(Image.fromarray(ex.pixel_values)) for ex in candidates] \
                for candidates in candidatess] \
                    for candidatess in candidatess_encodings
        ]
        examples['candidatess_num_patches'] = [
            [candidates[0].num_text_patches for candidates in candidatess] \
                for candidatess in candidatess_encodings
        ]
        examples['candidates_attention_mask'] = [
            [get_attention_mask(n) for n in num_patches] \
                for num_patches in examples['candidatess_num_patches']
        ]
        return examples
    
    def preprocess_examples(self, examples):
        try:
            preprocessed_examples = self._preprocess_examples(examples)
            self.prev_examples = preprocessed_examples
        except:
            pass
        return self.prev_examples


class MM_words_examples_preprocessor(examples_preprocessor):
    def __init__(
            self, renderer, words_sampler, candidates_creator, transforms, tokenizer, datasets, rtl=True
        ):
        super().__init__(renderer, words_sampler, candidates_creator, transforms, rtl)
        
        self.tokenizer = tokenizer
        self.datasets = datasets
        
    def custom_preprocessing(self, text):
        processed_text = ""
        for char in text:
            if char.isspace():
                processed_text += char
            elif len(self.tokenizer.tokenize(char)) == 0:
                processed_text += f" {self.tokenizer.unk_token} "
            else:
                processed_text += char
        return processed_text
    
    def _preprocess_examples(self, ds_sent_word_tuples):
        """
        Preprocess a batch of sentences and specific words in them, and applying transforms.
        
        :ds_sent_word_tuples: list of tuples which contain -
            dataset index - 0 for training, 1 for eval and 2 for test,
            sentence index - index of the sentence in the dataset,
            word index - index of the word in the split sentence
        """
        # extract the sentences
        examples = dict()
        ds_idx = ds_sent_word_tuples['d'][0]
        examples["text"] = [self.datasets[ds_idx][s]['text'] for s in ds_sent_word_tuples['s']]
        
        # render raw examples into images
        raw_text = [remove_diacritics(example) for example in examples["text"]]
        # replace unknown chars to [UNK], for keeping same number of words in text and raw_text
        raw_text = [self.custom_preprocessing(rt) for rt in raw_text]
        
        # for each example, sample words to train on
        indicess = [[w] for w in ds_sent_word_tuples['w']]
        wordss = [[ex.split()[indicess[i][0]]] for i, ex in enumerate(examples["text"])]

        # create labels
        labelss = torch.randint(
            low=0,
            high=self.candidates_creator.num_candidates,
            size=(len(examples["text"]), 1)
        )
        
        # for each sampled word, suggest some diacritizations and render them
        candidatess_lists = []
        candidatess_encodings = []
        for i, words in enumerate(wordss): # (sentences, num_words, num_candidates)
            candidatess_lists.append([])
            candidatess_encodings.append([])
            for j, word in enumerate(words):
                candidates = self.candidates_creator.get_candidates(word)
                
                # # todo check - MM_words_examples_preprocessor
                # raw_word = remove_diacritics(word)
                # # candidates[-1] = raw_word
                # candidates[-2], candidates[-1] = raw_word, raw_word
                
                # shuffle the wrong candidates
                candidates[1:] = random.sample(candidates[1:], len(candidates)-1)
                
                # put the correct candidates where the labels pointing on by rotation
                candidates = candidates[-labelss[i][j]:] + candidates[:-labelss[i][j]]
                candidatess_lists[i].append(candidates)
                candidates_encodings = [
                    # self.renderer(cand) for cand in candidates
                    self.apply_renderer(cand) for cand in candidates
                ]
                candidatess_encodings[i].append(candidates_encodings)

        # insert the preprocessed data to the batch dictionary
        examples['raw_text'] = raw_text
        examples['indicess'] = indicess
        examples['wordss'] = wordss
        examples['labelss'] = labelss
        examples['candidatess_lists'] = candidatess_lists
        examples['candidates_examples'] = [
            [[self.transforms(Image.fromarray(ex.pixel_values)) for ex in candidates] \
                for candidates in candidatess] \
                    for candidatess in candidatess_encodings
        ]
        examples['candidatess_num_patches'] = [
            [candidates[0].num_text_patches for candidates in candidatess] \
                for candidatess in candidatess_encodings
        ]
        examples['candidates_attention_mask'] = [
            [get_attention_mask(n) for n in num_patches] \
                for num_patches in examples['candidatess_num_patches']
        ]
        return examples
    
    def preprocess_examples(self, examples):
        try:
            preprocessed_examples = self._preprocess_examples(examples)
            self.prev_examples = preprocessed_examples
        except:
            pass
        return self.prev_examples


class contrastive_pairs_preprocessor:
    def __init__(
            self,
            renderer,
            candidates_creator,
            transforms,
            rtl=True
        ):
        self.renderer = renderer
        self.candidates_creator = candidates_creator
        self.transforms = transforms
        self.rtl = rtl
        self.prev_examples = None

    def apply_renderer(self, text):
        encoding = self.renderer(text)
        if self.rtl:
            encoding = examples_preprocessor.flip_pixel_values(encoding)
        return encoding
    
    def _preprocess_examples(self, pairs):
        """
        Preprocess a batch of diacritized word pairs that have the same diacritization.
        """
        
        # create raw words and raw encodings
        diac_words = []
        for pair in pairs["pairs"]:
            diac_words.extend([example for example in pair])
        raw_words = []
        for pair in pairs["pairs"]:
            raw_words.extend([remove_diacritics(example) for example in pair])
        raw_encodings = []
        for word in raw_words:
            raw_encodings.append(self.apply_renderer(word))
        
        # create labels
        labelss = torch.randint(
            low=0,
            high=self.candidates_creator.num_candidates,
            size=(len(raw_words), 1)
        )
        
        # for each sampled word, suggest some diacritizations and render them
        candidates_list = []
        candidate_encodings = []
        for i, pair in enumerate(pairs["pairs"]):
            for j, word in enumerate(pair):
                candidates = self.candidates_creator.get_candidates(word)
                
                # shuffle the wrong candidates
                candidates[1:] = random.sample(candidates[1:], len(candidates)-1)
                
                # put the correct candidates where the labels pointing on by rotation
                candidates = candidates[-labelss[i*2+j][0]:] + candidates[:-labelss[i*2+j][0]]
                candidates_list.append(candidates)
                candidates_encodings = [
                    self.apply_renderer(cand) for cand in candidates
                ]
                candidate_encodings.append(candidates_encodings)

        # insert the preprocessed data to the batch dictionary
        examples = dict()
        examples['diac_words'] = diac_words
        examples['raw_words'] = raw_words
        examples['raw_examples'] = \
            [self.transforms(Image.fromarray(example.pixel_values)) \
                for example in raw_encodings]
        examples['raw_num_patches'] = \
            [example.num_text_patches for example in raw_encodings]
        examples['raw_attention_mask'] = \
            [get_attention_mask(n) for n in examples['raw_num_patches']]
        examples['labelss'] = labelss
        examples['candidates_list'] = candidates_list
        examples['candidate_examples'] = \
            [[self.transforms(Image.fromarray(ex.pixel_values)) for ex in candidates] \
                for candidates in candidate_encodings]
        examples['candidate_num_patches'] = \
            [[candidate.num_text_patches for candidate in candidates] \
                for candidates in candidate_encodings]
        examples['candidate_attention_masks'] = [
            [get_attention_mask(n) for n in num_patches] \
                for num_patches in examples['candidate_num_patches']
        ]
        return examples
    
    def preprocess_examples(self, examples):
        try:
            preprocessed_examples = self._preprocess_examples(examples)
            self.prev_examples = preprocessed_examples
        except:
            pass
        return self.prev_examples
