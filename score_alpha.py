from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pickle
import random

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")


def tokenizer(sentence):
    sentence = word_tokenize(sentence, language='russian')
    sentence = list(filter(lambda x: x.isalnum() and x not in russian_stopwords, sentence))
    sentence = list(map(lambda x: morph.parse(x)[0].normal_form, sentence))
    return sentence

if True:
    texts = []
    with open('data/upd_v5.txt') as f:
        for i, line in enumerate(f):
            line = line.strip()
            texts.append(line)


    vec = TfidfVectorizer(tokenizer=tokenizer)
    vec.fit(texts)
    with open('vec.pickle', 'wb') as f:
        pickle.dump(vec, f)
else:
    with open('vec.pickle', 'rb') as f:
        vec = pickle.load(f)

pr_s = None
sims = []
pairs = []
with open('neg_alpha.txt') as f:
    for line in f:
        if pr_s is not None:
            vec1 = vec.transform([line.strip()])
            vec2 = vec.transform([pr_s])
            s = cosine_similarity(vec1, vec2)[0]
            if 0.2 < s:
                sims.append(s)
                pairs.append((pr_s, line.strip()))
        pr_s = line.strip()


x = zip(sims, pairs)
x = sorted(x, key=lambda y:y[0])
sims, pairs = [i for i, j in x], [j for i, j in x]

with open('sims_neg.txt', 'w') as f:
    for i, s in enumerate(sims):
        print(s, file=f)
        print(pairs[i][0], file=f)
        print(pairs[i][1], file=f)
        print(file=f)


print(max(sims), min(sims))
