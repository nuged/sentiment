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

texts = defaultdict(list)

wc = defaultdict(set)
lc = defaultdict(int)

with open('data/upd_v3.txt') as f:
    for i, line in enumerate(f):
        sentence, targets = line.strip().split('\t')
        sentence = sentence.strip()
        targets = targets.split('|')
        targets = list(map(lambda x: x.split(','), targets))
        for t in targets:
            label = t[0]
            lc[label] += 1
            start = int(t[1])
            end = int(t[2])
            w = sentence[start:end]
            for word in lexemes:
                if w in lexemes[word]:
                    wc[word].add(i)

vals = [len(wc[w]) for w in wc]
print('max\t', np.max(vals))
print('min\t', np.min(vals))
print('mean\t', np.mean(vals))
print('median\t', np.median(vals))
print('total\t', np.sum(vals))

print(lc)