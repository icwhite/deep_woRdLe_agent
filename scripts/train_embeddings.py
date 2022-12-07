from utils.embeddings.letters_to_words import LetterToWord
import numpy as np
import torch.nn.functional as Fun
import utils.infrastructure.pytorch_utils as ptu

wordle_words = open("scripts/wordle_words.txt", "r").read().split(",")
wordle_words = [word.replace('\n', '') for word in wordle_words]
wordle_words = np.array(wordle_words)

encoder = dict(zip(list("abcdefghijklmnopqrstuvwxyz"), np.arange(26)))
model = LetterToWord(n_letters=5, n_possible_words=len(wordle_words), n_layers=2, size=128)

def encode(word: str):
    """
    Inputs a word of length n_letters as a string and maps to a list of ints corresponding to
    each letter a = 0, b = 1, ..., z = 25
    """
    return [encoder[letter] for letter in word]

def train(model, iterations, batch_size):
    # sample batch size iterations from the wordle words
    # shuffle wordle words

    w_words = np.random.choice(wordle_words, len(wordle_words))


    for iter in range(iterations):

        word_indices = np.random.choice(len(wordle_words), batch_size)
        letters = np.array([encode(wordle_words[word_idx]) for word_idx in word_indices])
        letters = ptu.from_numpy(letters)

        one_hot_letters = Fun.one_hot(letters, num_classes=26).flatten(start_dim=1)
        one_hot_words = Fun.one_hot(word_indices, num_classes=len(wordle_words)).flatten(start_dim=1)

        loss = model.update(one_hot_letters, one_hot_words)

# def evaluation(model, batch_size):


# next do eval portion



