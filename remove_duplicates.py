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



exit(0)

idxs = defaultdict(list)

for i, t in enumerate(texts[:-1]):
    if i % 1000 == 0:
        print(i, '...')
    for j, s in enumerate(texts[i+1:]):
        if t == s:
            idxs[i].append(j)

with open('idxs.txt', 'w') as f:  # X is similar to X + 1 + Y
    for i in sorted(idxs):
        cols = idxs[i]
        cols = map(str, cols)
        f.write(str(i) + '\t' + ','.join(cols) + '\n')

exit(255)

print(datetime.datetime.now().time(), '\tvectorizing...')

vec = TfidfVectorizer(tokenizer=tokenizer)
X = vec.fit_transform(texts)

save_npz('vecs.npz', X)

X = load_npz('vecs.npz')


print(datetime.datetime.now().time(), '\tcos...')

idxs = {}

for i in range(X.shape[0] - 1):
    if i % 1000 == 0:
        print('proccessing {}-th step'.format(i))
    idx = []
    for j in range(i+1, X.shape[0]):
        if (X[i] != X[j]):
            continue
        idx.append(j)
    if idx:
        idxs[i] = idx



with open('idxs.txt', 'w') as f:  # X is similar to X + 1 + Y
    for i in sorted(idxs):
        cols = idxs[i]
        cols = cols.tolist()
        cols = map(str, cols)
        f.write(str(i) + '\t' + ','.join(cols) + '\n')

print(datetime.datetime.now().time(), '\tFINISH')
