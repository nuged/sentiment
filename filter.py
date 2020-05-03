from collections import  defaultdict


nouns = defaultdict(list)

with open('rusentilex_2017.txt') as f:
    for line in f:
        if line.startswith('!') or line == '\n':
            continue
        line = line.strip().split(', ')
        if line[1] == 'Noun':
            nouns[line[0]].append(line[3])


files = ['pos_nouns.txt', 'neg_nouns.txt']
pos = []
neg = []

for l, file in zip([pos, neg], files):
    with open(file) as f:
        for word in f:
            word = word.strip()
            l.append(word)
print('pos:')
for word in pos:
    if len(nouns[word]) > 1:
        print(word, nouns[word])

print('neg')
for word in neg:
    if len(nouns[word]) > 1:
        print(word, nouns[word])