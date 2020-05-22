from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pymorphy2
from random import sample, seed

morph = pymorphy2.MorphAnalyzer()

wc = defaultdict(set)
lc = defaultdict(int)
pc = defaultdict(set)
nc = defaultdict(set)

with open('data/upd_v5.txt') as f:
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
            w = morph.parse(w)[0].normal_form
            wc[w].add(i)
            if label == 'pos':
                pc[w].add(i)
            else:
                nc[w].add(i)

vals = [len(wc[w]) for w in wc]
pvals = [len(pc[w]) for w in pc]
nvals = [len(nc[w]) for w in nc]

print('max\t', np.max(vals))
print('min\t', np.min(vals))
print('mean\t', np.mean(vals))
print('median\t', np.median(vals))
print('total\t', np.sum(vals))

print(lc)

print('POS--------')
print('max\t', np.max(pvals))
print('min\t', np.min(pvals))
print('mean\t', np.mean(pvals))
print('median\t', np.median(pvals))
print('total\t', np.sum(pvals))


print('NEG----------')
print('max\t', np.max(nvals))
print('min\t', np.min(nvals))
print('mean\t', np.mean(nvals))
print('median\t', np.median(nvals))
print('total\t', np.sum(nvals))

plt.xticks(np.arange(0, 130, 10))
plt.hist(vals, bins=30)
plt.show()

plt.xticks(np.arange(0, 130, 10))
plt.hist(pvals, bins=30)
plt.show()

plt.hist(nvals, bins=20)
plt.show()
