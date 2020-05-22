pos = []
neg = []

with open('data/upd_v5.txt') as f:
    for line in f:
        line = line.strip()
        s, t = line.split('\t')
        cat, _, _ = t.split(',')
        if cat == 'neg':
            neg.append(s)
        else:
            pos.append(s)

pos = sorted(pos)
neg = sorted(neg)

for f, t in zip(['pos_alpha.txt', 'neg_alpha.txt'], [pos, neg]):
    with open(f,'w') as g:
        g.write('\n'.join(t))
