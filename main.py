import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import islice


class myDataset(Dataset):
    def __init__(self, file):
        self.file = file
        with open(file) as f:
            for i, l in enumerate(f):
                pass
        self.length = i + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with open(self.file) as f:
            line = next(islice(f, idx, idx + 1)).strip()
        line, labels = line.split('\t')
        labels = labels.split('|')
        for i, _ in enumerate(labels):
            labels[i] = labels[i].split(',')
            if len(labels[i]) != 3:
                raise IndexError('wrong label:\t{}'.format(labels[i]))
            if labels[i][0] not in ['pos', 'neg']:
                raise ValueError('incorrect label type:\t{}'.format(labels[i][0]))
            labels[i][1] = int(labels[i][1])
            labels[i][2] = int(labels[i][2])

        return line, labels


class Classifier(nn.Module):
    def __init__(self, bert_path, num_cats=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path, cache_dir=None)
        num_outputs =  num_cats
        self.fc = nn.Linear(768, num_outputs)

    def forward(self, x):
        x = self.bert(x, output_all_encoded_layers=False)[0]
        return self.fc(x)


def replace(text, subtext, start, end):
    return "".join((text[:start], subtext, text[end:]))

labels_to_idx = {'pos' : 1, 'neg' : 0}

def get_tensors(batch, tokenizer, max_length=512):
    texts, labels, starts, ends = batch
    s = []
    masked_labels = []
    for i, text in enumerate(texts):
        line = text
        for j, label in enumerate(labels[i]):
            line = '[CLS] ' + replace(line, ' [MASK] ', starts[i][j], ends[i][j]) + ' [SEP]'
        tokens = tokenizer.tokenize(line)
        diff = max_length - len(tokens)
        if diff > 0:
            tokens.extend(['[PAD]']*diff)
        elif diff < 0:
            tokens = tokens[:max_length]
        c = 0
        cur_ml = []
        for tok in tokens:
            if tok == '[MASK]':
                cur_ml.append(labels_to_idx[labels[i][c]])
                c += 1
            else:
                cur_ml.append(-1)
        s.append(tokenizer.convert_tokens_to_ids(tokens))
        masked_labels.append(cur_ml)
    return torch.LongTensor(s), torch.LongTensor(masked_labels)



def collate_fn(batch):
    lines = []
    labels = []
    starts = []
    ends = []
    for line, item in batch:
        lines.append(line)
        g_l = []
        g_e = []
        g_s = []
        for label, start, end in item:
            g_l.append(label)
            g_e.append(end)
            g_s.append(start)
        labels.append(g_l)
        starts.append(g_s)
        ends.append(g_e)
    return lines, labels, starts, ends


ds = myDataset('result_1.txt')
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

tok = BertTokenizer.from_pretrained('weights/ru/')
cls = Classifier('weights/ru/')

loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

with torch.no_grad():
    for i, batch in enumerate(dl):
        tokens, labels = get_tensors(batch, tok)
        output = cls(tokens)
        print(output.size(), labels.size())
        print(loss_fct(output.view(-1, 2), labels.view(-1)))

        exit(0)
