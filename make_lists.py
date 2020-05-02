import pymorphy2
morph = pymorphy2.MorphAnalyzer()

pos = set()
neg = set()
for file in ['pos_nouns.txt', 'neg_nouns.txt']:
    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            for parse in morph.parse(line):
                if 'anim' not in parse.tag:
                    continue
                for lex in parse.lexeme:
                    if file.startswith('pos_nouns.txt'):
                        pos.add(lex.word)
                    else:
                        neg.add(lex.word)

for name, cat in [('pos_full.txt', pos), ('neg_full.txt', neg)]:
    with open(name, 'w') as f:
        for word in sorted(cat):
            f.write(word + '\n')
