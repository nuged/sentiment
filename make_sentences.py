from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import os
from string import punctuation
from flashtext.keyword import KeywordProcessor
from nltk.corpus import stopwords
import nltk
import time
import spacy
import re
from multiprocessing import Pool


def find_all_indexes(input_str, search_str):
    search_str = re.escape(search_str)
    return [m.start() for m in re.finditer(search_str, input_str)]

nlp = spacy.load('ru2')
nltk.download('stopwords')
russian_stopwords = stopwords.words("russian")
stemmer = SnowballStemmer(language='russian')

pos_nouns = defaultdict(int)
neg_nouns = defaultdict(int)

for file in ['neg_full.txt', 'pos_full.txt']:
    with open(file, 'r') as f:
        for word in f.readlines():
            word = word.strip()
            if file.startswith('neg'):
                neg_nouns[word] = 1
            else:
                pos_nouns[word] = 1

proc = KeywordProcessor()
for word in pos_nouns:
    proc.add_keyword(stemmer.stem(word))
for word in neg_nouns:
    proc.add_keyword(stemmer.stem(word))

def check(text):
    return proc.extract_keywords(text)

def do(infile, outfile):
    with open(outfile, 'w') as f:
        for root, dirs, files in os.walk(infile):
            if not files:
                print('---processing {}---'.format(root))
                continue

            for file in files:
                if not file.endswith('.txt'):
                    continue
                path = os.path.join(root, file)
                with open(path, 'r') as source:
                    text = source.read()
                    text = text.split('\n')
                    for paragraph in text:
                        sentences = sent_tokenize(paragraph, language='russian')
                        for sentence in sentences:
                            if not (check(sentence)):
                                continue

                            tokens = []

                            for tok in word_tokenize(sentence):
                                if tok not in russian_stopwords and tok not in punctuation:
                                    tokens.append(tok)
                            if len(tokens) < 5:
                                continue

                            if sentence.startswith('- ') or sentence.startswith('– '):
                                sentence = sentence[2:]

                            found = []
                            for name, category in [('pos', pos_nouns), ('neg', neg_nouns)]:
                                for i, token in enumerate(tokens):
                                    if token in category:
                                        found.extend([(tokens[i], name)])

                            labels = []
                            for pair in found:
                                idxs = find_all_indexes(sentence, pair[0])
                                for idx in idxs:
                                    labels.extend([(pair[1], str(idx), str(idx + len(pair[0])))])

                            if found:
                                labels = list(map(lambda x: ','.join(x), labels))
                                f.write(sentence + '\t' + '|'.join(labels) + '\n')


if __name__ == '__main__':
    pool = Pool(3)
    pool.starmap(do, [('data/201101','result_1.txt'),
                  ('data/201102', 'result_2.txt'),
                  ('data/201103', 'result_3.txt')])