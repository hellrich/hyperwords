from __future__ import print_function
from docopt import docopt
import random
import copy
from collections import defaultdict
from representations.representation_factory import create_representation
import numpy as np
from scipy.stats.stats import spearmanr
import sys
import os.path

def main():
    args = docopt("""
        Usage:
            eval_reliability.py [options] <representation> <file_name> <folders>...

        Options:
            --words FILE      Use FILE with list of words (1 per line) to measure reliabilty
            --ws FILES        Testsets for word similarity evaluation, use "," as separator!
            --ana FILES       Testsets for analogy evaluation, use "," as separator!
            --closest N       Use N closest neighbors to measure reliability [default: 10]   
    """)
    folders = args["<folders>"]

    closest = int(args["--closest"])
    word_list = args["--words"]
    ws_test_sets = [read_ws_test_set(path) for path in args["--ws"].split(",")]
    as_test_sets = [read_as_test_set(path) for path in args["--ana"].split(",")]
    as_xi_and_ix = [get_vocab_as(test_set) for test_set in as_test_sets]
    words = words_to_evaluate_file(word_list) if word_list else argswords_to_evaluate(representations)


    #good default parameter for svd
    args["--eig"] = 0
    args["--w+c"] = False
    #not used
    args["--neg"] = 1

    representations = []
    for file in folders:
        if os.path.isfile(file+"/"+args["<file_name>"]+".words.vocab"):
            x = copy.deepcopy(args)
            x["<representation_path>"] = file+"/"+args["<file_name>"]
            representations.append(create_representation(x))
        else:
            print("Could not find "+file+"/"+args["<file_name>"]+".words.vocab", file=sys.stderr)
    #comparisson over all subsets
    if len(representations) < 2:
        raise Exception("Need multiple models for evaluation")

    evaluated = [" ".join([str(evaluate_ws(r,w)) for r in representations]) for w in ws_test_sets]
    for i, test_set in enumerate(as_test_sets):
        evaluated.append(" ".join([str(evaluate_as(r,test_set, as_xi_and_ix[i][0], as_xi_and_ix[i][1])) for r in representations]))
    evaluated.append(reliability(representations, words, closest))
    print("\t".join(evaluated))
    


def reliability(representations, words, closest):
    results = []
    for i in range(0,len(representations)):
        results.append(0)
    for word in words: #list(words)[:5]:#
        neighbors = [get_neighbors(representation, word, closest) for representation in representations]

        #comparisson over all subsets
        for i in range(0,len(representations)):
            results[i] += jaccard(neighbors[:i] + neighbors[i+1:])
    for i in range(0,len(representations)):
        results[i] /= len(words) 
    return " ".join([str(r) for r in results])

def jaccard(sets):
    if (len(sets) < 2):
        raise Exception("Need multiple sets")
    for s in sets:
        if len(s) == 0:
            return 0

    intersection = copy.copy(sets[0])
    for s in sets[1:]:
        intersection &= s

    union = set()
    for s in sets:
        union |= s

    return (1.0 * len(intersection))/len(union)

def words_to_evaluate_file(filename):
    words = set()
    with open(filename) as f:
        for line in f:
            words.add(line.strip())
    return words

def words_to_evaluate(representations):
    words = representations[0].wi.viewkeys()
    for r in representations[1:]:
        words &= r.wi.viewkeys()
    return words

def get_neighbors(representation, word, closest):
    if word in representation.wi:
        dist = representation.m.dot(representation.m[representation.wi[word]].T)
        dist[representation.wi[word]] = -np.Inf
        return {representation.iw[x] for x in np.argsort(-dist)[:closest]}
    else:
        return set()

def evaluate_ws(representation, data):
    results = []
    for (x, y), sim in data:
        results.append((representation.similarity(x, y), float(sim)))
    actual, expected = zip(*results)
    return spearmanr(actual, expected)[0]

def read_ws_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            x, y, sim = line.strip().lower().split()
            test.append(((x, y), sim))
    return test

def read_as_test_set(path):
    test = []
    with open(path) as f:
        for line in f:
            analogy = line.strip().lower().split()
            test.append(analogy)
    return test 

def evaluate_as(representation, data, xi, ix):
    sims = prepare_similarities_as(representation, ix)
    correct_mul = 0.0
    for a, a_, b, b_ in data:
        b_mul = guess(representation, sims, xi, a, a_, b)
        if b_mul == b_:
            correct_mul += 1
    return correct_mul/len(data)

#vocab = ix
def prepare_similarities_as(representation, vocab):
    vocab_representation = representation.m[[representation.wi[w] if w in representation.wi else 0 for w in vocab]]
    sims = vocab_representation.dot(representation.m.T)
    
    dummy = None
    for w in vocab:
        if w not in representation.wi:
            dummy = representation.represent(w)
            break
    if dummy is not None:
        for i, w in enumerate(vocab):
            if w not in representation.wi:
                vocab_representation[i] = dummy
    
    if type(sims) is not np.ndarray:
        sims = np.array(sims.todense())
    else:
        sims = (sims+1)/2
    return sims


def guess(representation, sims, xi, a, a_, b):
    sa = sims[xi[a]]
    sa_ = sims[xi[a_]]
    sb = sims[xi[b]]
    
    mul_sim = sa_*sb*np.reciprocal(sa+0.01)
    if a in representation.wi:
        mul_sim[representation.wi[a]] = 0
    if a_ in representation.wi:
        mul_sim[representation.wi[a_]] = 0
    if b in representation.wi:
        mul_sim[representation.wi[b]] = 0
    return representation.iw[np.nanargmax(mul_sim)]

def get_vocab_as(data):
    vocab = set()
    for analogy in data:
        vocab.update(analogy)
    vocab = sorted(vocab)
    return dict([(a, i) for i, a in enumerate(vocab)]), vocab

if __name__ == "__main__":
    main()
