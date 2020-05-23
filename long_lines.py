import numpy as np
import matplotlib.pyplot as plt

c = 0
with open('data/upd_v5.txt') as f, open('data/dataset.txt', 'w') as g:
    for line in f:
        if len(line) < 500:
            g.write(line)


