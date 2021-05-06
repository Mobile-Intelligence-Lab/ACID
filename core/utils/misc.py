import numpy as np
import torch

import os
import sys
from datetime import datetime

root_path = os.path.dirname(os.path.realpath(__file__)) + '/../../logs'
sys.path.append(root_path)


class Dot(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def log(msg, filename='output.txt'):
    if type(msg) in (list, tuple):
        for message in msg:
            log(message, filename)
    else:
        with open(root_path + '/' + filename, mode='a+') as f:
            f.write(f"[{str(datetime.now())}] {msg}\n")


def log_error(msg, filename='error.txt'):
    log(msg, filename)


def dist(x, y):
    if len(x.shape) == 1:
        x = x.view(-1, 1)
    if len(y.shape) == 1:
        y = y.view(-1, 1)

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    d = torch.pow(x - y, 2).sum(2)
    return d


def extend_dataset(model, df, cats, label_tag):
    cols = df.columns.values.tolist()
    rf_features = []

    for i in range(model.kernels_size):
        cols.append(f'kernel_feature_{i}')
    cols.append(label_tag)

    print(f"Creating extended dataset...")
    for i, row in df.iterrows():
        rf_features_tmp = row.tolist()
        x = torch.FloatTensor(np.array([rf_features_tmp]))
        attack_type = cats[i]
        with torch.no_grad():
            label = model(x)[0]
        label = label.max(dim=1).indices
        label = label[0][0].item()

        features = model.sub_nets[label].kernel_weights

        rf_features_tmp.extend(features.squeeze().tolist())
        rf_features_tmp.extend([attack_type])

        rf_features.append(rf_features_tmp)
    print(f"Done creating extended dataset")

    return rf_features, cols
