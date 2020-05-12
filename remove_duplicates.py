from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pymorphy2
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict


def tokenizer(sentence):
    sentence = word_tokenize(sentence, language='russian')
    sentence = list(filter(lambda x: x.isalnum() and x not in russian_stopwords, sentence))
    sentence = list(map(lambda x: morph.parse(x)[0].normal_form, sentence))
    return sentence

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")
texts = set()

print(datetime.datetime.now().time(), '\tReading data...')

with open('data/updated.txt') as f:
    for i, line in enumerate(f):
        line = line.strip()
        texts.add(line)

with open('data/updated.txt') as f, open('data/upd_v2.txt', 'w') as g:
    for i, line in enumerate(f):
        line = line.strip()
        if line in texts:
            texts.remove(line)
            g.write(line + '\n')
