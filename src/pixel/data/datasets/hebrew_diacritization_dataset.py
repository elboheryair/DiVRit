# class SamplesCreator:
#     """
#     Transforms a diacritized sentence into an input for the model
#     """
#
#     def __call__(self, diacritized_sentence):
#         diacritized_sentence = ''.join(c for c in diacritized_sentence if c in HEBREW_LETTERS_AND_DIACRITICS_AND_SPACE)
#         raw_sentence = ''.join(c for c in diacritized_sentence if c in HEBREW_LETTERS_AND_SPACE)
#         sentences_with_one_diacritized_word = self._get_sentences_with_one_diacritized_word(raw_sentence,
#                                                                                       diacritized_sentence)
#         return raw_sentence, sentences_with_one_diacritized_word
#
#     def _get_sentences_with_one_diacritized_word(self, raw_sentence, diacritized_sentence):
#         one_diacritized_word_in_sentences = []
#         raw_sentence_spaces = [0] + [idx.start() for idx in re.finditer(' ', raw_sentence)] + [len(raw_sentence)]
#         diacritized_sentence_spaces = [0] + [idx.start() for idx in re.finditer(' ', diacritized_sentence)] + [
#             len(diacritized_sentence)]
#         for i in range(len(raw_sentence_spaces) - 1):
#             begin = raw_sentence[:raw_sentence_spaces[i]]
#             middle = diacritized_sentence[diacritized_sentence_spaces[i]:diacritized_sentence_spaces[i + 1]]
#             end = raw_sentence[raw_sentence_spaces[i + 1]:]
#             i_word_diacritized_in_sentence = begin + middle + end
#             one_diacritized_word_in_sentences.append(i_word_diacritized_in_sentence)
#         return one_diacritized_word_in_sentences
#
#
# class HebrewDiacritizationDataset(Dataset):
#     def __init__(self, diacritized_sentences, transform=SamplesCreator()):
#         self.diacritized_sentences = diacritized_sentences
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.diacritized_sentences)
#
#     def __getitem__(self, idx):
#         diacritized_sentence = self.diacritized_sentences[idx]
#         raw_sentence, sentences_with_one_diacritized_word = self.transform(diacritized_sentence)
#         return raw_sentence, sentences_with_one_diacritized_word




# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TextIO, Union


import torch
from filelock import FileLock
from PIL import Image
from transformers import PreTrainedTokenizer, is_torch_available


from ...utils import Modality, Split, get_attention_mask
from ..rendering import PyGameTextRenderer, PangoCairoTextRenderer


logger = logging.getLogger(__name__)


@dataclass
class HebrewDiacritizationInputExample:
    """
    A single training/test example for hebrew diacritization

    Args:
        guid: Unique id for the example.
        idx: index of the word to diacritize
        sent: undiacritized sentence
        cands: sentences with one diacritized word, this word is in the specified
        idx. The first candidate is the correct one.
        Specified for train and dev examples, but not for test examples.
    """

    guid: str # needed for the cross entropy ignore_index, as padding label id
    idx: int
    sent: str
    cands: List[str]
    # labels: Optional[List[List[List[int]]]] # todo change type, probably it needs to be the diacritized chars


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    class HebrewDiacritizationDataset(Dataset):
        """
        Dataset for hebrew diacritization.
        """

        features: List[Dict[str, Union[int, torch.Tensor]]]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            processor: Union[Union[PyGameTextRenderer, PangoCairoTextRenderer], PreTrainedTokenizer],
            labels: List[str],
            transforms: Optional[Callable] = None,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.TRAIN,
            **kwargs,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode.value, processor.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.examples = read_examples_from_file(data_dir=data_dir, mode=mode)
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    self.examples = read_examples_from_file(data_dir=data_dir, mode=mode)
                    self.features = convert_examples_to_image_features(
                       self.examples, labels, max_seq_length, processor, transforms, **kwargs
                    )
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> Dict[str, Union[int, torch.Tensor]]:
            return self.features[i]


def line_split_chars_and_labels(line):
    chars, labels = [], []
    for c in line:
        if is_hebrew_letter(c):
            chars.append(c)
            labels.append([])
        elif is_hebrew_niqqud(c):
            labels[-1].append(c)
        else:
            chars.append(c)
            labels.append([])
    return chars, labels


def is_shuruq(chars, labels, i, j):
    return chars[i] == 'ו' and \
        labels[i][j] == '\u05bc' and \
        len(labels[i]) == 1


def get_niqqud_classification(chars, labels, i, j):
    is_vowel, is_shin, is_dagesh = False, False, False
    cur_niqqud = labels[i][j]
    if cur_niqqud in NIQQUD_SIN[1:]:
        is_shin = True
    elif is_shuruq(chars, labels, i, j):
        is_vowel = True
    elif cur_niqqud in DAGESH[1:]:
        is_dagesh = True
    else:
        is_vowel = True
    return is_vowel, is_shin, is_dagesh


def get_one_hot_from_indices_list(indices_label):
    one_hot_label = [
        [0] * len(NIQQUD),
        [0] * len(NIQQUD_SIN),
        [0] * len(DAGESH)
    ]
    one_hot_label[0][indices_label[0]] = 1
    one_hot_label[1][indices_label[1]] = 1
    one_hot_label[2][indices_label[2]] = 1
    return one_hot_label


def map_line_to_one_hot_labels(chars, labels):
    one_hot_labels = []
    for i, lst in enumerate(labels):
        indices_label = [0, 0, 0]
        for j in range(len(lst)):
            is_vowel, is_shin, is_dagesh = get_niqqud_classification(chars, labels, i, j)
            niqqud = lst[j]
            if is_vowel:
                indices_label[0] = NIQQUD.index(niqqud)
            elif is_shin:
                indices_label[1] = NIQQUD_SIN.index(niqqud)
            elif is_dagesh:
                indices_label[2] = DAGESH.index(niqqud)
        one_hot_label = get_one_hot_from_indices_list(indices_label)
        one_hot_labels.append(one_hot_label)
    return one_hot_labels


def read_examples_from_file(data_dir, mode: Union[Split, str], label_idx=-1) -> List[HebrewDiacritizationInputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.txt")

    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                continue
            line = line [:-1] if line[-1] == "\n" else line
            splits = line.split("<>")
            idx = splits[0]
            sent = splits[1]
            cands = splits[2:]
            examples.append(HebrewDiacritizationInputExample(guid=f"{mode}-{guid_index}", idx=idx, sent=sent, cands=cands))
            guid_index += 1


            # chars, labels = line_split_chars_and_labels(line)
            # # assert len(chars) == len(labels)
            # labels = map_line_to_one_hot_labels(chars[i], labels[i])
            # examples.append(HebrewDiacritizationInputExample(guid=f"{mode}-{guid_index}", chars=chars, labels=labels))
            # guid_index += 1
    return examples


def write_predictions_to_file(writer: TextIO, test_input_reader: TextIO, preds_list: List):
    example_id = 0
    for line in test_input_reader:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            writer.write(line)
            if not preds_list[example_id]:
                example_id += 1
        elif preds_list[example_id]:
            output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
            writer.write(output_line)
        else:
            logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])


def get_labels(path: str) -> List[List[List[int]]]:
    # every character gets vector in length 3 as diacritizing instructions, 90 options:
    # 15 - RAFE, SHVA, REDUCED_SEGOL, REDUCED_PATAKH, REDUCED_KAMATZ, HIRIK, TZEIRE,
    # SEGOL, PATAKH, KAMATZ, HOLAM, KUBUTZ, SHURUK, METEG, ['\u05b7'].
    # 2 - RAFE, DAGESH.
    # 3 - RAFE, SHIN_YEMANIT, SHIN_SMALIT.
    return [
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0], [0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1], [1, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1], [0, 1]]
    ]


def convert_examples_to_image_features(
    examples: List[HebrewDiacritizationInputExample],
    label_list: List[str],
    max_seq_length: int,
    processor: Union[PyGameTextRenderer, PangoCairoTextRenderer],
    transforms: Optional[Callable] = None,
    pad_token_label_id=-100,
) -> List[Dict[str, Union[int, torch.Tensor, List[int]]]]:
    """Loads a data file into a list of `Dict` containing image features"""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info(f"Writing example {ex_index} of {len(examples)}")

        encoding = processor(example.words)
        image = encoding.pixel_values
        num_patches = encoding.num_text_patches
        word_starts = encoding.word_starts

        label_ids = [pad_token_label_id] * max_seq_length
        for idx, word_start in enumerate(word_starts[:-1]):
            label_ids[word_start] = label_map[example.labels[idx]]

        pixel_values = transforms(Image.fromarray(image))
        attention_mask = get_attention_mask(num_patches, seq_length=max_seq_length)


        # sanity check lengths
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info(f"sentence: {' '.join(example.words)}")
            logger.info(f"attention_mask: {attention_mask}")
            logger.info(f"label_ids: {label_ids}")

        features.append({"pixel_values": pixel_values, "attention_mask": attention_mask, "label_ids": label_ids})

    return features
