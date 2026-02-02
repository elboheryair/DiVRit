import numpy as np
np.random.seed(42) #TODO: Erase
import json
from pixel.utils.candidates_creators.candidates_creators_manager import (
    remove_diacritics,
    remove_non_hebrew_chars
)


class words_sampler:
    """ sample words to train the diacritizer on """

    def __init__(self, sampling_func):
        """
        :sampling_func: words sampling algorithm
        """
        self.sampling_func = sampling_func
    
    def sample_words(self, sentences):
        indicess = []
        wordss = []
        for sentence in sentences:
            indices, words = self.sampling_func(sentence)
            indicess.append(indices)
            wordss.append(words)
        return indicess, wordss


class uniform_sampling:
    """
    function that samples words from a sentence uniformly. If the number of words to
    to sample is bigger than the number of words in the sentence, there will be
    repetitions on words
    """

    def __init__(self, num_words=1):
        """
        :num_words: number of words to train on from a given sentence
        """
        self.num_words = num_words

    def uniform_dist_sampler(self, sentence):
        """ sample words from the given sentence uniformly """
        all_words = sentence.split()
        if len(all_words) < self.num_words: # sample with repetitions
            indices = np.random.choice(len(all_words), self.num_words)
        else: # no repetitions
            indices = np.arange(len(all_words))
            np.random.shuffle(indices)
            indices = indices[:self.num_words]
        indices.sort()
        sampled_words = [all_words[idx] for idx in indices]
        return indices, sampled_words

    def __call__(self, sentence):
        return self.uniform_dist_sampler(sentence)


class log_frequency_sampling:
    """
    function that samples words from a sentence, where the probability of i'th
    word to be chosen is log of it's number of appearances in the data. The
    sampling is not repetative, as long as possible.
    """

    def __init__(self, words_counter, num_words=1):
        """
        :num_words: number of words to train on from a given sentence
        :words_counter: counter of the raw words appearances in the data
        """
        self.num_words = num_words
        self.words_counter = words_counter
    
    def sample_words(self, words, counters, to_replace):
        """
        Samples words indices from an array based on the log probabilities of their counters. Returns
        a list of sampled words indices.
        :words: array of words to sample from.
        :counters: array of counters corresponding to each word.
        :to_replace: sample with repetitions if True, else without repetitions.
        """
        if len(words) != len(counters):
            raise ValueError("The words and counters arrays must have the same length.")

        # Compute log probabilities
        log_probs = np.log2(counters)

        # # Convert log probabilities to probabilities
        # # Add a small constant to counters to avoid issues with log(0)
        # probs = np.exp(log_probs - np.max(log_probs)) # Subtract max(log_probs) for numerical stability
        # probs /= probs.sum()  # Normalize to sum to 1
        log_probs /= log_probs.sum()

        # Sample words based on the computed probabilities
        # indices = np.random.choice(len(words), size=self.num_words, p=probs, replace=to_replace)
        indices = np.random.choice(len(words), size=self.num_words, p=log_probs, replace=to_replace)
        return indices

    def __call__(self, sentence):
        all_words = sentence.split()
        raw_words = [remove_non_hebrew_chars(w) for w in all_words]
        counters = [1 for _ in raw_words]
        for i, w in enumerate(raw_words):
            if w in self.words_counter:
                counters[i] += self.words_counter[w]
        # counters = [self.words_counter[w] for w in all_words]
        to_replace = len(all_words) < self.num_words # if True, sample with repetitions
        indices = self.sample_words(all_words, counters, to_replace)
        indices.sort()
        sampled_words = [all_words[idx] for idx in indices]
        return indices, sampled_words


class adaptive_sampling:
    """
    Function that samples words from a sentence. Initially, the probability of each
    word to be chosen is log of it's number of appearances in the data. The 
    probability mass of a word becomes higher when the model predicts wrong
    candidate, and lower when the prediction is correct. The sampling is not
    repetative, as long as possible.
    """

    def __init__(self, words_counter, num_words=1):
        """
        :num_words: number of words to train on from a given sentence
        :words_counter: counter of the raw words appearances in the data. If the
        training is a resume of previous run, the counter can be a path to the
        counter from the previous run, with the existing probability mass
        """
        self.num_words = num_words
        self.words_counter = words_counter
        if isinstance(words_counter, dict):
            # update the probability mass
            for key, val in self.words_counter.items():
                self.words_counter[key] = np.log2(val + 1)
        else:
            # load dictionary from the JSON file
            with open(words_counter, "r", encoding="utf-8") as f:
                self.words_counter = json.load(f)

    def good_func(self, x):
        """ return a decreased probability mass, with a fixed point in 1.0 """
        return 1 + ((x-1) ** 1.038) * 0.89
    
    def bad_func(self, x):
        """ return an increased probability mass, with a fixed point in 16.01 """
        new_x = np.max([17-x, 1.0])
        return 17 - self.good_func(new_x) * 0.99
    
    def sample_words(self, words, counters, to_replace):
        """
        Samples words indices from an array based on the log probabilities of their
        counters. Returns list with the indices of the sampled words.
        :words: array of words to sample from.
        :counters: array of counters corresponding to each word.
        :to_replace: sample with repetitions if True, else without repetitions.
        """
        if len(words) != len(counters):
            raise ValueError("The words and counters arrays must have the same length.")

        # Compute log probabilities
        counters = np.array(counters)
        log_probs = counters / counters.sum()

        # Sample words based on the computed probabilities
        # indices = np.random.choice(len(words), size=self.num_words, p=probs, replace=to_replace)
        indices = np.random.choice(len(words), size=self.num_words, p=log_probs, replace=to_replace)
        return indices

    # def update_counters(self, words, good_preds, bad_preds):
    def update_counters(self, wordss, probs, labelss):
        """
        Get the raw words that the model diacritized, and update their counters
        based on it's success
        """
        # create boolean tensor with good and bad predictions
        predictions = probs.argmax(1)
        hits = predictions == labelss
        words = sum(wordss, [])

        for idx, hit in enumerate(hits):
            word = remove_non_hebrew_chars(words[idx])
            if word not in self.words_counter:
                continue
            cur = self.words_counter[word]
            # print(f"before updating: {idx}, {word}, {hit}, {cur}")
            if hit:
                nxt = self.good_func(cur)
                self.words_counter[word] = nxt
            else:
                nxt = self.bad_func(cur)
                self.words_counter[word] = nxt
            # print(f"after updating:  {idx}, {word}, {hit}, {nxt}")
    
    def __call__(self, sentence):
        all_words = sentence.split()
        raw_words = [remove_non_hebrew_chars(w) for w in all_words]
        counters = [0 for _ in raw_words]
        for i, w in enumerate(raw_words):
            if w in self.words_counter:
                counters[i] += self.words_counter[w]
        # counters = [self.words_counter[w] for w in all_words]
        to_replace = len(all_words) < self.num_words # if True, sample with repetitions
        indices = self.sample_words(all_words, counters, to_replace)
        indices.sort()
        sampled_words = [all_words[idx] for idx in indices]
        return indices, sampled_words
