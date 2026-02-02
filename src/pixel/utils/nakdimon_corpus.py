import os
import random
random.seed(42) # ERASE
from os import path
from datasets import Dataset, DatasetDict, concatenate_datasets
import re
import numpy as np
np.random.seed(42) #TODO: Erase


root_dir = path.abspath('../nakdimon/')

nakdimon_training_corpus = {
    'premodern': tuple([
        'hebrew_diacritized/poetry',
        'hebrew_diacritized/rabanit',
        'hebrew_diacritized/pre_modern',
        'hebrew_diacritized/shortstoryproject_predotted'
    ]),
    'automatic': tuple([
        'hebrew_diacritized/shortstoryproject_Dicta',
    ]),
    'modern': tuple([
        'hebrew_diacritized/modern',
        'hebrew_diacritized/dictaTestCorpus',
    ])
}

nakdimon_validation_corpus = {
    'validation': tuple([
        'hebrew_diacritized/validation/modern',
    ]),
}

nakdimon_test_corpus = {
    'test': tuple([
        'hebrew_diacritized/test_modern',
    ]),
}


letters_diacritics_pattern = '[^א-ת\u0591-\u05C7]'
diacritics_pattern = re.compile(r'[\u0591-\u05C7]')
letter_list = '[ א-ת]' # hebrew letters and space
threshold = 1800


def iterate_files(base_paths, root_dir=root_dir):
    """ extract all the paths of text files in base_paths """
    for name in base_paths:
        full_path = os.path.join(root_dir, name)
        # if not os.path.isdir(name):
        if not os.path.isdir(full_path):
            yield name
            continue
        # for root, dirs, files in os.walk(name):
        for root, dirs, files in os.walk(full_path):
            for fname in files:
                path = os.path.join(root, fname)
                # path = os.path.join(name, fname)
                yield path


def pad_non_hebrew_words(sentence):
    """
    In order to use visual represnetation for word level tasks, we need to identify
    the patches in which the word appears. Then, we can use the embeddings of the
    patches for our task.
    Spaces are, generally, represented as 4 blank columns, but in many cases (for
    instance in latin scripts) the words are seperated with less then 4 columns. This
    dataset is predominantly in Hebrew, so we add one space before and after each
    non-Hebrew word to assure that we will recognize all the words in the image later.
    """

    # Regular expression to match Hebrew characters, diacritics and punctuation
    hebrew_regex = re.compile(r'[\u0590-\u05FF\uFB1D-\uFB4F]+')
    
    # Split the sentence into words, keeping the delimiters
    words = re.findall(r'\S+|\s+', sentence)
    
    for i, word in enumerate(words):
        if re.search(hebrew_regex, word):
            continue
        # Surround non-Hebrew words with spaces, if not already surrounded
        if not word.startswith(' ') and not word.endswith(' '):
            words[i] = f' {word} '
        elif word.startswith(' ') and not word.endswith(' '):
            words[i] = f'{word} '
        elif not word.startswith(' ') and word.endswith(' '):
            words[i] = f' {word}'
    
    return ''.join(words)


def length_without_diacritics(line):
    """ approximation of the length of the undicritized line length """
    count = len(re.findall(letter_list, line))
    return count


def load_data_for_pixel_pretraining2(base_paths):
    """
    Iterate all the text files in the given directory path, and return a list of all the
    examples in them. Each example is a concatenation of several rows from one file, and
    the length of the examples is limited according to a fixed threshold
    """

    # extract all the paths of text files
    corpora = [filename for filename in iterate_files(base_paths)]
    assert False not in [f[-4:] == '.txt' for f in corpora]
    
    # read the data into a list of "long" strings
    corpus = []
    for file in corpora:
        len_sum = 0
        accumulated = ""
        with open(os.path.join(root_dir, file), 'r') as f:
            for line in f.readlines():
                if line.strip() == "":
                    continue
                if line[-1] == "\n":
                    line = line[:-1]
                cur_len = length_without_diacritics(line)
                if cur_len >= threshold:
                    corpus.append(pad_non_hebrew_words(line)) # append line
                elif len_sum + cur_len >= threshold * 1.1:
                    if len_sum > cur_len:
                        corpus.append(pad_non_hebrew_words(accumulated)) # append accumulated
                        len_sum = cur_len
                        accumulated = line
                    else:
                        corpus.append(pad_non_hebrew_words(line)) # append line
                elif len_sum + cur_len >= threshold:
                    accumulated += " " + line
                    corpus.append(pad_non_hebrew_words(accumulated))
                    len_sum = 0
                    accumulated = ""
                else:
                    len_sum += cur_len
                    accumulated += " " + line
            if len_sum > 0:
                corpus.append(pad_non_hebrew_words(accumulated))
    return corpus


def find_nth_non_diacritic_char_index(text):
    count = 0
    for char in text:
        # Check if the character is not a Hebrew diacritic
        if ord(char) < 0x0591 or ord(char) > 0x05C7:
            count += 1
            if count == threshold:
                return count
    return None

def split_text_by_threshold(text):
    segments = []
    new_threshold = find_nth_non_diacritic_char_index(text)
    while new_threshold != None:
        # Find the index to split after the threshold
        split_index1 = text[new_threshold:].find('. ')
        split_index2 = text[new_threshold:].find('\n')
        if split_index1 > -1 and split_index2 > -1:
            split_index = min(split_index1, split_index2)
        else:
            split_index = max(split_index1, split_index2)
        if split_index == -1:
            break
        split_index += new_threshold
        segments.append(text[:split_index + 1].strip())
        text = text[split_index+1:].strip()
        new_threshold = find_nth_non_diacritic_char_index(text)
    if len(text) > 0:
        segments.append(text)
    return segments


def load_data_for_pixel_pretraining(base_paths):
    """
    Iterate all the text files in the given directory path, and return a list of all the
    examples in them. Each example is a concatenation of several rows from one file, and
    the length of the examples is limited according to a fixed threshold
    """

    # extract all the paths of text files
    corpora = [filename for filename in iterate_files(base_paths)]
    assert False not in [f[-4:] == '.txt' for f in corpora]
    
    # read the data into a list of "long" strings
    corpus = []
    for file in corpora:
        with open(os.path.join(root_dir, file), 'r') as f:
            text = f.read()
            lines = split_text_by_threshold(text)
            lines = [pad_non_hebrew_words(line.replace('\n', ' ').strip()) \
                     for line in lines]
            lines = [pad_hebrew_separators(line) for line in lines]
            corpus.extend(lines)

    return corpus


def load_data(base_paths):
    """
    Iterate all the text files in the given directory path, and return a list of all the
    examples in them. Each example is a single row
    """

    # extract all the paths of text files
    corpora = [filename for filename in iterate_files(base_paths)]
    assert False not in [f[-4:] == '.txt' for f in corpora]
    
    # read the data into a list
    corpus = []
    for file in corpora:
        with open(os.path.join(root_dir, file), 'r') as f:
            # text = f.read()
            # if text[-1] != "\n":
            #     text += "\n"
            # corpus.append(text)
            for line in f.readlines():
                if line.strip() != "":
                    corpus.append(line)
            if line[-1] != '\n':
                corpus[-1] += '\n'
    
    return corpus
    

def get_dataset(corpus, pixel_pretraining=False):
    """ load the data from the files and create the Dataset """

    # logging.info("Loading training data...")
    data_dict = {}
    candidates_dict = dict()
    counters = dict()
    for stage_name, stage_dataset_filenames in corpus.items():
        # logging.info(f"Loading training data: {stage_name}...")
        if pixel_pretraining:
            split = load_data_for_pixel_pretraining(stage_dataset_filenames)
        else:
            split = load_data(stage_dataset_filenames)
        extend_dict(candidates_dict, counters, split)
        random.shuffle(split) # shuffle in place
        split = [{'text': sample} for sample in split]
        data_dict[stage_name] = split
    
    # new_dataset = DatasetDict(data_dict)
        
    for key, val in data_dict.items():
        data_dict[key] = Dataset.from_list(val)
    
    new_dataset = DatasetDict(data_dict)

    return new_dataset, candidates_dict, counters


def create_dataset(pixel_pretraining=False):
    """
    Load the data in the diacritized hebrew corpus of Pinter & Gershuny, and return it as
    a Dataset. The splits are fit the name of the stage (i.e. category) of the texts in
    it (e.g. 'premodern' split).
    Intermediate pretraining of a Pixel model requires the examples to be longer, for the
    sake of pretraining all the parameters of the model. If pixel_pretraining is True
    the examples from the same diacritized document are concatenated, until their total
    length get to a fixed threshold. The default threshold is 1800, and it can be changed
    using the set_threshold function.
    """

    print("creating Nakdimon dataset")
    
    # Set the random seed for reproducibility
    random.seed(42)

    # create the datasets
    train_dataset, train_candidates_dict, train_counters = get_dataset(
        corpus=nakdimon_training_corpus, pixel_pretraining=pixel_pretraining)
    validation_dataset, validation_candidates_dict, validation_counters = get_dataset(
        corpus=nakdimon_validation_corpus, pixel_pretraining=pixel_pretraining)

    return train_dataset, validation_dataset, train_candidates_dict, \
        validation_candidates_dict, train_counters


def set_threshold(new_threshold=1800):
    """ default global threshold is 1800, this function allow changing it """
    global threshold
    threshold = new_threshold


def remove_hebrew_diacritics(text):
    # Substitute diacritics with an empty string
    return diacritics_pattern.sub('', text)


def extend_dict(candidates_dict, counters, text):
    """
    Gets candidates dictionary, counters dictionary and list of strings, adds all
    the candidates from the strings to the candidates dictionary and updates the
    counters
    """
    
    for line in text:
        for word in line.split():
            word = re.sub(letters_diacritics_pattern, '', word)
            raw_word = remove_hebrew_diacritics(word)
            if word == raw_word:
                continue
            if raw_word in candidates_dict:
                candidates_dict[raw_word].add(word)
                counters[raw_word] += 1
            else:
                candidates_dict[raw_word] = {word}
                counters[raw_word] = 1


def create_toy_dataset(pixel_pretraining=False):
    
    print("creating toy dataset")

    # Set the random seed for reproducibility
    random.seed(42)

    # load data into lists
    toy_data_path = "diacritics/oscar_heb/nakdimon_toy.txt"
    split = []
    with open(toy_data_path, 'r') as f:
        text = f.read()
        lines = split_text_by_threshold(text)
        lines = [pad_non_hebrew_words(line.replace('\n', ' ').strip()) for line in lines]
        split.extend(lines)
    candidates_dict = dict()
    counters = dict()
    extend_dict(candidates_dict, counters, split)
    random.shuffle(split) # shuffle in place
    limit = int(len(split) * 0.8)
    train_split, eval_split = split[:limit], split[limit:]

    # create dictionaries of datasets from the splits
    train_dict = {}
    split = [{'text': sample} for sample in train_split]
    train_dict['train'] = Dataset.from_list(split)
    split = [{'text': sample} for sample in eval_split]
    eval_dict = {}
    eval_dict['eval'] = Dataset.from_list(split)
    
    return train_dict['train'], eval_dict['eval'], candidates_dict, counters


####################################################
#################  Words Dataset   #################
####################################################


def get_words_dataset(corpus, pixel_pretraining=False):
    """ load the data from the files and create the Dataset """

    # logging.info("Loading training data...")
    data_dict = {}
    for stage_name, stage_dataset_filenames in corpus.items():
        # logging.info(f"Loading training data: {stage_name}...")
        if pixel_pretraining:
            split = load_data_for_pixel_pretraining(stage_dataset_filenames)
        else:
            split = load_data(stage_dataset_filenames)
        random.shuffle(split) # shuffle in place
        split = [{'text': sample} for sample in split]
        data_dict[stage_name] = split
    
    filter_threshold = np.min([1800, 4 * threshold])
    for key, val in data_dict.items():
        # Create dataset split
        cur_data = Dataset.from_list(val)
        # Filter out too long and problematic examples
        cur_data = cur_data.filter(lambda x: (len(x["text"]) < filter_threshold))
        cur_data = cur_data.filter(lambda x: "\u200b" not in x["text"]) # \u200b is zero-width space
        data_dict[key] = cur_data
    
    new_dataset = DatasetDict(data_dict)

    return new_dataset


def get_words_test_dataset(corpus, pixel_pretraining=False):
    """ load the data from the files and create the Dataset """

    data_dict = {}
    for stage_name, stage_dataset_filenames in corpus.items():
        if pixel_pretraining:
            split = load_data_for_pixel_pretraining(stage_dataset_filenames)
        else:
            split = load_data(stage_dataset_filenames)
        # random.shuffle(split) # shuffle in place
        # split = [{'text': sample} for sample in split]
        split = [{'text': sample.replace("\u200b", "")} for sample in split]
        data_dict[stage_name] = split
    
    # filter_threshold = np.min([1800, 4 * threshold])
    for key, val in data_dict.items():
        # Create dataset split
        cur_data = Dataset.from_list(val)
        # Filter out too long and problematic examples
        # cur_data = cur_data.filter(lambda x: (len(x["text"]) < filter_threshold))
        # cur_data = cur_data.filter(lambda x: "\u200b" not in x["text"]) # \u200b is zero-width space
        data_dict[key] = cur_data
    
    new_dataset = DatasetDict(data_dict)

    return new_dataset


def create_words_dataset(pixel_pretraining=False):
    """
    Load the data in the diacritized hebrew corpus of Pinter & Gershuny, and return it as
    a Dataset. The splits are fit the name of the stage (i.e. category) of the texts in
    it (e.g. 'premodern' split).
    Intermediate pretraining of a Pixel model requires the examples to be longer, for the
    sake of pretraining all the parameters of the model. If pixel_pretraining is True
    the examples from the same diacritized document are concatenated, until their total
    length get to a fixed threshold. The default threshold is 1800, and it can be changed
    using the set_threshold function.
    """

    print("creating Nakdimon words dataset")
    
    # Set the random seed for reproducibility
    random.seed(42)

    # create the datasets
    train_dataset = get_words_dataset(corpus=nakdimon_training_corpus, pixel_pretraining=pixel_pretraining)
    train_dataset = concatenate_datasets([
        train_dataset['premodern'],
        train_dataset['automatic'],
        train_dataset['modern']]
    )
    
    # create the dictionaries
    candidates_dict = dict()
    train_appearances_dict = dict()
    extend_cands_and_appearances_dicts(
        candidates_dict, train_appearances_dict, train_dataset, 0
    )
    validation_dataset = get_words_dataset(corpus=nakdimon_validation_corpus, pixel_pretraining=pixel_pretraining)
    validation_dataset = validation_dataset["validation"]
    validation_appearances_dict = dict()
    extend_cands_and_appearances_dicts(
        # candidates_dict, validation_appearances_dict, validation_dataset, 1
        dict(), validation_appearances_dict, validation_dataset, 1
    )
    test_dataset = get_words_test_dataset(corpus=nakdimon_test_corpus, pixel_pretraining=pixel_pretraining)
    test_dataset = test_dataset["test"]
    test_appearances_dict = dict()
    extend_cands_and_appearances_dicts(
        dict(), test_appearances_dict, test_dataset, 2
    )

    return train_dataset, candidates_dict, train_appearances_dict, validation_dataset, \
           validation_appearances_dict, test_dataset, test_appearances_dict


def extend_cands_and_appearances_dicts(
        candidates_dict, appearances_dicts, text, dataset_idx
    ):
    """
    Get candidates dictionary, appearances dictionary and list of strings, add all the
    candidates from the list to the candidates dictionary, updates the counters, and
    keep the sentence and word indeices where the word appears
    """
    
    for i, line in enumerate(text):
        for j, word in enumerate(line['text'].split()):
            word = re.sub(letters_diacritics_pattern, '', word)
            raw_word = remove_hebrew_diacritics(word)
            if word == raw_word:
                continue
            if raw_word in candidates_dict:
                candidates_dict[raw_word].add(word)
            else:
                candidates_dict[raw_word] = {word}
            if word in appearances_dicts:
                appearances_dicts[word].append((dataset_idx, i, j))
            else:
                appearances_dicts[word]= [(dataset_idx, i, j)]


# Regular expression to detect Hebrew words separated by a non-spacing character
HEBREW_WORD_SEPARATORS = re.compile(r'([\u05D0-\u05EA\u05B0-\u05BC\u05C1\u05C2\u05C7]+)([^ \t\'"\u05D0-\u05EA\u05B0-\u05BC\u05C1\u05C2\u05C7]+)([\u05D0-\u05EA\u05B0-\u05BC\u05C1\u05C2\u05C7]+)')

def pad_hebrew_separators(text):
    """
    Ensure that any non-spacing character between Hebrew words is padded with spaces.
    """
    return HEBREW_WORD_SEPARATORS.sub(r'\1 \2 \3', text)


def create_toy_words_dataset(pixel_pretraining=False):
    
    print("creating toy words dataset")

    # Set the random seed for reproducibility
    random.seed(42)

    # load data into lists
    toy_data_path = "diacritics/oscar_heb/nakdimon_toy.txt"
    split = []
    with open(toy_data_path, 'r') as f:
        text = f.read()
        lines = split_text_by_threshold(text)
        lines = [pad_non_hebrew_words(line.replace('\n', ' ').strip()) for line in lines]
        lines = [pad_hebrew_separators(line) for line in lines]
        split.extend(lines)
    random.shuffle(split) # shuffle in place
    
    # split = split[:20] # todo delete after implementing the callback!!
    # random.shuffle(split) # todo delete after implementing the callback!!

    limit = int(len(split) * 0.8)
    train_split, eval_split, test_split = split[:limit], split[limit:], split[limit:]

    # create dictionaries of datasets from the splits
    train_dict = {}
    train_split = [{'text': sample} for sample in train_split]
    train_dict['train'] = Dataset.from_list(train_split)
    eval_split = [{'text': sample} for sample in eval_split]
    eval_dict = {}
    eval_dict['eval'] = Dataset.from_list(eval_split)
    test_dict = {}
    test_split = [{'text': sample} for sample in test_split]
    test_dict['test'] = Dataset.from_list(test_split)

    # Filter out too long examples
    filter_threshold = np.min([1800, 4 * threshold])
    train_dataset = train_dict['train'].filter(lambda x: (len(x["text"]) < filter_threshold))
    eval_dataset = eval_dict['eval'].filter(lambda x: (len(x["text"]) < filter_threshold))
    test_dataset = test_dict['test'].filter(lambda x: (len(x["text"]) < filter_threshold))
    train_dataset = train_dict['train'].filter(lambda x: "\u200b" not in x["text"]) # \u200b is zero-width space
    eval_dataset = eval_dict['eval'].filter(lambda x: "\u200b" not in x["text"])
    test_dataset = test_dict['test'].filter(lambda x: "\u200b" not in x["text"])

    candidates_dict = dict()
    train_appearances_dict = dict()
    extend_cands_and_appearances_dicts(candidates_dict, train_appearances_dict, train_dataset, 0)
    eval_appearances_dict = dict()
    extend_cands_and_appearances_dicts(candidates_dict, eval_appearances_dict, eval_dataset, 1)
    
    test_appearances_dict = dict()
    extend_cands_and_appearances_dicts(dict(), test_appearances_dict, test_dataset, 2)

    # return train_dataset, candidates_dict, train_appearances_dict, eval_dataset, eval_appearances_dict
    return train_dataset, candidates_dict, train_appearances_dict, eval_dataset, \
           eval_appearances_dict, test_dataset, test_appearances_dict
