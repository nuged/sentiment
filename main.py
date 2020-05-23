import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import islice
from sklearn.metrics import confusion_matrix
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import KFold


class myDataset(Dataset):
    def __init__(self, file, idx=None):
        self.file = file
        with open(file) as f:
            self.content = f.read().split('\n')
        if idx is not None:
            self.content = [self.content[i] for i in idx]

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        line = self.content[idx]
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
        num_outputs = num_cats
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
    return torch.LongTensor(s).cuda(), torch.LongTensor(masked_labels).cuda()



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


def remove_ignored(pred, gt):
    idx = gt != -1
    pred = pred[idx]
    gt = gt[idx]
    return pred, gt


def get_metrics(pred, gt):
    conf = confusion_matrix(gt, pred, [0, 1])
    metrics = {}
    TP = conf[1, 1]
    TN = conf[0, 0]
    FP = conf[0, 1]
    FN = conf[1, 0]
    if TP + FP == 0:
        prec = 0
    else:
        prec = TP / (TP + FP)
    if TP + FP == 0:
        rec = 0
    else:
        rec = TP / (TP + FN)
    if prec == rec == 0:
        metrics['F1'] = 0
    else:
        metrics['F1'] = 2 * prec * rec / (prec + rec)
    metrics['precision'] = prec
    metrics['recall'] = rec
    metrics['accuracy'] = (TP + TN) / (TP + FP + TN + FN)
    return metrics



def train(model, loader, optimizer, loss_fct, number, val_loader=None, num_epochs=10, logfile=None):
    loss_history = []
    loss_val = []
    metrics_history = []
    running_loss = 0.0
    loss_track = []
    for epoch in range(num_epochs):
        loss_epoch = []
        for i, batch in enumerate(loader):
            texts, labels = get_tensors(batch, tok)
            optimizer.zero_grad()
            output = model(texts)
            loss = loss_fct(output.view(-1, 2), labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
                loss_epoch.append(running_loss / 10)
                loss_track.append(running_loss / 10)
                running_loss = 0.0
        loss_history.append(loss_epoch)
        plt.plot(loss_track)
        if val_loader is not None:
            metrics, loss = validate(model, val_loader, loss_fct)
            loss_val.append(loss)
            metrics_history.append(metrics)
            print('\n', metrics, '\n')
        torch.save(model.state_dict(), 'models/model-{}-ep-{}'.format(number, epoch))

    if logfile is not None:
        with open(logfile, 'w') as f:
            f.write('---val metrics\n')
            for m in metrics_history:
                f.write(json.dumps(m))
                f.write('\n')
            f.write('---val loss\n')
            for l in loss_val:
                f.write(str(l) + '\n')
            f.write('---train loss')
            for l in loss_track:
                f.write(str(l) + '\n')


def validate(model, loader, loss_fct):
    with torch.no_grad():
        y_pred = []
        y_true = []
        loss_history = []
        for i, batch in enumerate(loader):
            texts, labels = get_tensors(batch, tok)
            output = model(texts)
            loss = loss_fct(output.view(-1, 2), labels.view(-1))
            loss_history.append(loss.item())
            pred, labels = remove_ignored(output.argmax(dim=2), labels)
            y_pred.extend(pred.cpu().detach())
            y_true.extend(labels.cpu().detach())
    metrics = get_metrics(y_pred, y_true)
    return metrics, np.mean(loss_history)

tok = BertTokenizer.from_pretrained('weights/ru/')
loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

kf = KFold(n_splits=5, shuffle=True, random_state=23)
c = 0
for train_idx, test_idx in kf.split(range(18851)):
    torch.manual_seed(0)
    cls = Classifier('weights/ru/')
    cls.cuda()
    opt = optim.Adam(cls.parameters())
    train_ds = myDataset('data/dataset.txt', train_idx)
    test_ds = myDataset('data/dataset.txt', test_idx)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=1, collate_fn=collate_fn)
    train(cls, train_dl, opt, loss_fct, test_dl, c, logfile='logs/' + str(c) + '.txt')
    c += 1

