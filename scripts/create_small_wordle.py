import numpy as np

wordle_words = open("scripts/wordle_words.txt", "r").read().split(",")
wordle_words = [word.replace('\n', '') for word in wordle_words]

subset = np.random.choice(wordle_words, 100).tolist()

f = open("scripts/wordle_subset.txt", "w")

for items in subset:
    f.writelines(items + ",\n")

f.close()