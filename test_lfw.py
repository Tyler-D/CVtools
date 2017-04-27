#!/usr/bin/python
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
import argparse
import functools
import os


def l2_distance(x, y):
    return np.sum((x - y) ** 2, axis=1)


def cosine_distance(x, y):
    return 1 - np.sum(x * y, axis=1)


distance_methods = {
        "l2": l2_distance,
        "cosine": cosine_distance
}


def resovle_name(l):
    l = l.split()
    if len(l) == 3:
        n1 = "{}_{}".format(l[0], l[1].zfill(4))
        n2 = "{}_{}".format(l[0], l[2].zfill(4))
        is_same = True
    elif len(l) == 4:
        n1 = "{}_{}".format(l[0], l[1].zfill(4))
        n2 = "{}_{}".format(l[2], l[3].zfill(4))
        is_same = False
    return n1, n2, is_same


def get_embedings(feats, name2idx, pairs):
    e1s, e2s, ys = [], [], []
    for a, b, l in pairs:
        e1, e2 = name2idx[a], name2idx[b]
        e1s.append(e1)
        e2s.append(e2)
        y = 1 if l == 'True' else 0
        ys.append(y)
    return feats[e1s].copy(), feats[e2s].copy(), np.asarray(ys, np.bool)


def pca(txs, vxs, n_components=128):
    pca = PCA(n_components=n_components)
    pca.fit(np.concatenate(txs))

    result = []
    for x in txs + vxs:
        result.append(pca.transform(x))

    return result


def test(args, feats, name2idx, t_pairs, v_pairs):
    global distance_methods

    # get embedding
    tx1, tx2, tys = get_embedings(feats, name2idx, t_pairs)
    vx1, vx2, vys = get_embedings(feats, name2idx, v_pairs)

    if args.pca:
        # do pca
        tx1, tx2, vx1, vx2 = pca([tx1, tx2], [vx1, vx2], args.n_components)

    # compute distance
    distance_method = distance_methods[args.distance_method]
    tds = distance_method(tx1, tx2)
    vds = distance_method(vx1, vx2)

    # get best threholds
    threholds = np.arange(0, 4, 0.01)

    best_t, best_acc = 0, 0
    for t in threholds:
        predict = tds < t
        acc = (predict == tys).mean()
        if acc > best_acc:
            best_t = t
            best_acc = acc

    # compute test acc
    predict = vds < best_t
    acc = (predict == vys).mean()

    # get the wrong pairs
    wrong = v_pairs[predict != vys]

    return best_t, acc, wrong


def cross_validation(args, feats, name2idx, pairs, n_folds=10):
    N = len(pairs)
    fold_size = int(N / n_folds)
    idx_all = np.arange(N)

    accs = []
    ts = []
    ws = []
    for start in range(0, N, fold_size):
        v_idx = idx_all[start:start+fold_size]
        t_idx = np.hstack([idx_all[:start], idx_all[start+fold_size:]])

        t, acc, w = test(args, feats, name2idx, pairs[t_idx], pairs[v_idx])

        accs.append(acc)
        ts.append(t)
        ws.append(w)

    return accs, ts, ws


def read_data(args):

    # load features
    feats = np.loadtxt(args.features, dtype='float32', delimiter=args.sep)
    feats /= np.linalg.norm(feats, ord=2, axis=1, keepdims=True)

    # load labels
    with open(args.labels) as f:
        labels = f.readlines()
    labels = [os.path.splitext(os.path.basename(i))[0] for i in labels]

    # label to idx: {label: idx}
    name2idx = dict(zip(labels, range(len(feats))))

    # load pairs
    with open(args.pairs) as f:
        pairs = f.readlines()
    pairs = np.asarray(map(resovle_name, pairs[1:]))

    return feats, name2idx, pairs


def main(args):
    global distance_methods

    feats, name2idx, pairs = read_data(args)

    accs, ts, ws = cross_validation(args, feats, name2idx, pairs)

    # print test result
    print("Fold\tAcc\tThreshold")
    for i, (acc, t) in enumerate(zip(accs, ts)):
        print("{:d}\t{:.4f}\t{:.2f}".format(i, acc, t))
    print("acc: {:.4f} +/- {:.4f}".format(np.mean(accs), np.std(accs)))

    if args.save:
        np.savetxt(args.save, np.concatenate(ws)[:, :2], fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test lfw acc")
    parser.add_argument("features", help="features file")
    parser.add_argument("labels", help="labels file")
    parser.add_argument("pairs", help="pairs file")
    parser.add_argument("--pca", "-p", action="store_true", help="use pca, default: False")
    parser.add_argument("--save", "-f", default=None, help="save wrong list")
    parser.add_argument("--n_components", "-n", default=128, type=int, help="pca n components, default: 128")
    parser.add_argument("--distance_method", "-d",
                        choices=["l2", "cosine"], default="cosine",
                        help="distance type, default: cosine")
    parser.add_argument("--sep", "-s", default=",",
                        help="delimiter to use, default: ,")


    args = parser.parse_args()
    main(args)
