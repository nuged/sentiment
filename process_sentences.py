from collections import defaultdict
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#TODO: write new file

def find_all_indexes(input_str, search_str):
    search_str = re.escape(search_str)
    return [m.start() for m in re.finditer(search_str, input_str)]

russian_stopwords = stopwords.words("russian")
sent_tok = lambda x: sent_tokenize(x, language='russian')


def get_target_words(line, targets):
    targets = targets.split('|')
    targets = list(map(lambda x: x.split(','), targets))
    targets = {line[int(t[1]):int(t[2])]: t[0] for t in targets}
    return targets


def filter_by_length(sentences):
    new_list = []
    for sentence in sentences:
        tokens = []
        for tok in word_tokenize(sentence, language='russian'):
            if tok not in russian_stopwords and tok.isalnum():
                tokens.append(tok)
        if len(tokens) >= 10:
            new_list.append(sentence)
    return new_list

def get_positions(s2t):
    result = defaultdict(list)
    for sentence in s2t:
        for word in s2t[sentence]:
            idxs = find_all_indexes(sentence, word)
            if not idxs:
                continue
            for idx in idxs:
                result[sentence].append((s2t[sentence][word], str(idx), str(idx + len(word))))
    return result


def write(file, s2t):
    for s in s2t:
        targets = s2t[s]
        targets = list(map(lambda x: ','.join(x), targets))
        file.write(s + '\t' + '|'.join(targets) + '\n')

nouns = {}

for file in ['neg_full.txt', 'pos_full.txt']:
    with open(file, 'r') as f:
        for word in f.readlines():
            word = word.strip()
            if file.startswith('neg'):
                label = 'neg'
            else:
                label = 'pos'
            nouns[word] = label


def find_all(sentences, nouns):
    s2t = defaultdict(dict)
    for sentence in sentences:
        tokens = word_tokenize(sentence, language='russian')
        for t in tokens:
            if t in nouns:
                s2t[sentence].update({t: nouns[t]})
        if 'ЗА ДОЛГ ПО АЛИМЕНТАМ «НАСТУЧАТ»МИЛИЦИИ' in sentence:
            print(s2t)
    return s2t



with open('data/result.txt') as f, open('data/updated.txt', 'w') as g:
    for i, rline in enumerate(f):
        line, targets = rline.strip().split('\t')
        line = line.strip()

        new = re.sub(r'((?:[а-яa-z]\S|\s|\d)[\.!\?])([A-ZА-Я]|[«"\(])', '\g<1> \g<2>', line)

        if new != line:
            sentences = sent_tok(new)
            sentences = filter_by_length(sentences)
            sent_to_targets = find_all(sentences, nouns)
            sent_to_targets = get_positions(sent_to_targets)
            write(g, sent_to_targets)
        else:
            g.write(rline)
