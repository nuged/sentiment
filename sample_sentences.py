from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pymorphy2
from random import sample, seed

morph = pymorphy2.MorphAnalyzer()
lexemes = defaultdict(set)



for file in ['pos_nouns.txt', 'neg_nouns.txt']:
    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            for parse in morph.parse(line):
                if 'anim' not in parse.tag:
                    continue
                for lex in parse.lexeme:
                    lexemes[line].add(lex.word)


texts = defaultdict(set)
labels = {}

with open('data/upd_v2.txt') as f:
    for i, line in enumerate(f):
        sentence, targets = line.strip().split('\t')
        sentence = sentence.strip()
        targets = targets.split('|')
        targets = list(map(lambda x: x.split(','), targets))
        if len(targets) > 1:
            continue
        for t in targets:
            label = t[0]
            start = int(t[1])
            end = int(t[2])
            w = sentence[start:end]
            w = morph.parse(w)[0].normal_form
            texts[w].add(i)
            if w in labels:
                if labels[w] != label:
                    raise ValueError
            else:
                labels[w] = label

selected_sents = set()

seed(5)

for word in texts:
    num = len(texts[word])
    label = labels[word]
    if label == 'pos':
        maxnum = 120
    else:
        maxnum = 40
    if num > maxnum:
        cur = sample(texts[word], maxnum)
    else:
        cur = texts[word]
    selected_sents.update(cur)



with open('data/upd_v2.txt') as f, open('data/upd_v3.txt', 'w') as g:
    for i, line in enumerate(f):
        if i in selected_sents:
            g.write(line)

print(len(selected_sents))