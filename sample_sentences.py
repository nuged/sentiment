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

with open('data/upd_v2.txt') as f:
    for i, line in enumerate(f):
        sentence, targets = line.strip().split('\t')
        sentence = sentence.strip()
        targets = targets.split('|')
        targets = list(map(lambda x: x.split(','), targets))
        for t in targets:
            label = t[0]
            start = int(t[1])
            end = int(t[2])
            w = sentence[start:end]
            for word in lexemes:
                if w in lexemes[word]:
                    texts[word].add(i)

selected = {}
selected_sents = set()

seed(5)

for word in texts:
    num = len(texts[word])
    if num > 40:
        cur = sample(texts[word], 32)
    else:
        cur = texts[word]
    selected[word] = [i for i in cur if i not in selected_sents]
    selected_sents.update(cur)

with open('data/upd_v2.txt') as f, open('data/upd_v3.txt', 'w') as g:
    for i, line in enumerate(f):
        if i in selected_sents:
            g.write(line)

print(len(selected_sents))