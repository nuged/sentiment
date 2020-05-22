from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy2
import datetime


def tokenizer(sentence):
    sentence = word_tokenize(sentence, language='russian')
    sentence = list(filter(lambda x: x.isalnum() and x not in russian_stopwords, sentence))
    sentence = list(map(lambda x: morph.parse(x)[0].normal_form, sentence))
    return sentence

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

print(datetime.datetime.now().time(), '\tReading data...')

texts = []
with open('data/upd_v5.txt') as f:
    for i, line in enumerate(f):
        line = line.strip()
        texts.append(line)


print(datetime.datetime.now().time(), '\tvectorizing...')

vec = TfidfVectorizer(tokenizer=tokenizer)
X = vec.fit_transform(texts)



print(datetime.datetime.now().time(), '\tcos...')

print(X.shape)
idxs = {}

for i in range(X.shape[0] - 1):
    if i % 1000 == 0:
        print('proccessing {}-th step'.format(i))
    similarities = cosine_similarity(X[i], X[i+1:])
    idx = []
    for j, sim in enumerate(similarities[0]):
        if sim > 0.4:
            idx.append(j)
    if idx:
        idxs[i] = idx

for i in idxs:
    idxs[i] = list(map(lambda x: x + i + 1, idxs[i]))  # X is similar to X + 1 + Y

c = 0
for i in idxs:
    if c == 7:
        break
    main = texts[i]
    others = [texts[j] for j in idxs[i]]
    print('main:\t', main)
    print('similar sents:')
    for s in others:
        print(s)
    c += 1

if True:
    print(datetime.datetime.now().time(), '\tsaving indexes')
    with open('data/idxs.txt', 'w') as f:
        for i in sorted(idxs):
            cols = idxs[i]
            cols = map(str, cols)
            f.write(str(i) + '\t' + ','.join(cols) + '\n')

print(datetime.datetime.now().time(), '\tFINISH')
