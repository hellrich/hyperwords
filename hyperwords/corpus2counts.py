from __future__ import print_function
from collections import Counter
from math import sqrt, fabs
from random import Random
from docopt import docopt
import sys



def main():
    args = docopt("""
    Usage:
        corpus2counts.py [options] <corpus>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
        --dw         Use dynamic window (exclusive with --ww)
        --ww         Use weighted window (exclusive with --dw)
        --psub NUM   Probabilistic subsampling threshold (exclusive with --dsub) [default: 0]
        --dsub NUM   Deterministic subsampling threshold (exclusive with --psub) [default: 0]
        --pairs FILE Stores pairs in FILE [default: False]
        --out FILE   Stores count output in FILE [default: False]
        --debug FILE Store debugging info in FILE [default: False]
    """)

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    pairs = open_if_not_false(args["--pairs"])
    out = open_if_not_false(args["--out"])
    debug = open_if_not_false(args["--debug"])
    win = int(args['--win'])
    dynamic_window = args["--dw"]
    weighted_window = args["--ww"]
    if dynamic_window and weighted_window:
        raise Exception("Dynamic and weighted window options are exclusive!")
    psubsample = float(args['--psub']) #probabilistic
    dsubsample = float(args['--dsub']) #deterministic
    if psubsample and dsubsample:
        raise Exception("Subsampling options are exclusive!")


    vocab = read_vocab(corpus_file, thr, debug)
    corpus_size = sum(vocab.itervalues())
    if psubsample:
        subsample = psubsample * corpus_size
    elif dsubsample:
        subsample = dsubsample * corpus_size
    if psubsample or dsubsample:
        #changed to prob from 1-prob !
        subsampler = {word : sqrt(subsample / count) for word, count in vocab.iteritems() if count > subsample}


    rnd = Random()
    counts = Counter()
    with open(corpus_file) as f:
        for line in f:
            if debug:
                print("Processing4counts: "+line, file=debug)
            #dirty subsampling, results in vales lower than dsub!
            tokens = [t if t in vocab else None for t in line.strip().split()]
            if psubsample:
                tokens = [t if t not in subsampler or rnd.random() <= subsampler[t] else None for t in tokens]
            len_tokens = len(tokens)
            for i, tok in enumerate(tokens):
                if tok is not None:
                    if dynamic_window:
                        offset = rnd.randint(1, win)
                    else:
                        offset = win
                    start = i - offset
                    if start < 0:
                        start = 0
                    end = i + offset + 1
                    if end > len_tokens:
                        end = len_tokens
                    for j in xrange(start, end): 
                        if j != i and tokens[j] is not None:
                            if weighted_window:
                                distance = fabs(i-j)
                                count = (win + 1 - distance) / win
                            else:
                                count = 1
                            if dsubsample:
                                tok1_factor = subsampler[tok] if tok in subsampler else 1
                                tok2_factor = subsampler[tokens[j]] if tokens[j] in subsampler else 1
                                count = tok1_factor * tok2_factor * count
                            if pairs:
                                print(str(count)+" "+ tok + " " + tokens[j], file=pairs)
                                #pairs.write(str(count)+" "+ tok + " " + tokens[j]+"\n")
                            counts[(tok, tokens[j])] += count
    if not out:
        out = file=sys.stdout #somehow not working properly?
    for pair, count in counts.iteritems():
        print(str(count)+" "+pair[0] + " " + pair[1], file=out)

def open_if_not_false(name):
    if name == None:
        return False
    return open(name, "w")

def read_vocab(corpus_file, thr, debug=False):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            if debug:
                print("Processing4vocab: "+line, file=debug)
            vocab.update(Counter(line.strip().split()))
    if debug:
        print("Types: "+str(len(vocab)), file=sys.stderr)
    vocab = {token : count for token, count in vocab.iteritems() if count >= thr}
    if debug:
        print("Types over threshold: "+str(len(vocab)), file=sys.stderr)
    return vocab


if __name__ == '__main__':
    main()
