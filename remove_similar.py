from collections import defaultdict

written = set()
texts = []

with open('data/upd_v5.txt') as f:
    for i, line in enumerate(f):
        line = line.strip()
        texts.append(line)

idxs = defaultdict(list)

with open('data/idxs.txt') as f:
    for line in f:
        i, js = line.strip().split('\t')
        js = js.split(',')
        idxs[int(i)] = list(map(int, js))

with open('data/upd_v5.txt', 'w') as g:
    for i, line in enumerate(texts):
        if i not in idxs and i not in written:
            g.write(line + '\n')
        elif i not in written:
            g.write(line + '\n')
            written.update([i] + idxs[i])
