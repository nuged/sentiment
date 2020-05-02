pos, neg = set(), set()
pos1, neg1 = set(), set()

def read(file, s):
    with open(file) as f:
        for line in f:
            s.add(line.strip())

read('pos_nouns.txt', pos)
read('neg_nouns.txt', neg)
read('pos_nouns.txt', pos1)
read('neg_nouns.txt', neg1)

print(pos.intersection(neg))