import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

fold_metrics = defaultdict(list)
fold_loss = defaultdict(list)

for i in range(5):
    filename = 'logs/' + str(i) + '.txt'
    with open(filename) as f:
        for line in f:
            if line.strip() == '':
                continue
            if line.startswith('---val metr'):
                get_val_metr = True
                get_val_loss = False
                continue
            if line.startswith('---val loss'):
                get_val_loss = True
                get_val_metr = False
                continue
            if line.startswith('---train'):
                break
            if get_val_metr:
                metr = json.loads(line)
                fold_metrics[i].append(metr)
            elif get_val_loss:
                loss = float(line.strip())
                fold_loss[i].append(loss)

pr = np.zeros((5, 4)) # 5 folds and 4 epochs
rec = np.zeros((5, 4))
f1 = np.zeros((5, 4))
acc = np.zeros((5, 4))
loss = np.zeros((5, 4))

for fold in fold_metrics:
    metr_epochs = fold_metrics[fold]
    for i, metric in enumerate(metr_epochs):
        pr[fold, i] = metric['precision']
        rec[fold, i] = metric['recall']
        f1[fold, i] = metric['F1']
        acc[fold, i] = metric['accuracy']

for fold in fold_loss:
    for i, val in enumerate(fold_loss[fold]):
        loss[fold, i] = val

print('precision')
print(pr.mean(axis=0) * 100)

print('rec')
print(rec.mean(axis=0) * 100)

print('F1')
print(f1.mean(axis=0) * 100)

print('accuracy')
print(acc.mean(axis=0) * 100)

print('loss')
print(loss.mean(axis=0))

for m in zip(['precision', 'recall', 'F1', 'accuracy', 'loss (CE)'], [pr, rec, f1, acc, loss]):
    title, m = m
    if m is not loss:
        m *= 100
    plt.plot(m.mean(axis=0))
    plt.grid()
    plt.title(title)
    plt.show()