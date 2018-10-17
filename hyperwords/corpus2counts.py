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
        --pairs FILE  Stores pairs in FILE [default: False]
        --debug      Print debugging info to STDERR
    """)

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    pairs= args["--pairs"]
    if pairs == "False":
        pairs = False
    else:
        pairs=open(pairs, "w")
    win = int(args['--win'])
    dynamic_window = args["--dw"]
    weighted_window = args["--ww"]
    debug=args["--debug"]
    if dynamic_window and weighted_window:
        raise Exception("Dynamic and weighted window options are exclusive!")
    psubsample = float(args['--psub'])
    dsubsample = float(args['--dsub'])
    if psubsample and dsubsample:
        raise Exception("Subsampling options are exclusive!")


    vocab = read_vocab(corpus_file, thr, debug)
    corpus_size = sum(vocab.values())
    if psubsample:
        subsample = psubsample * corpus_size
    elif dsubsample:
        subsample = dsubsample * corpus_size
    if psubsample or dsubsample:
        #changed to prob from 1-prob !
        subsampler = dict([(word, sqrt(subsample / count)) for word, count in vocab.items() if count > subsample])


    rnd = Random()
    counts = Counter()
    with open(corpus_file) as f:
        if debug:
            print("Creating counts")
        for line in f:
            if debug:
                print("Processing: "+line, file=sys.stderr)
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
                                pairs.write(str(count)+" "+ tok + " " + tokens[j]+"\n")
                            counts[(tok, tokens[j])] += count
    for pair, count in counts.items():
        print(str(count)+" "+pair[0] + " " + pair[1])


def read_vocab(corpus_file, thr, debug=False):
    vocab = Counter()
    with open(corpus_file) as f:
        if debug:
            print("Creating vocabulary")
        for line in f:
            if debug:
                print("Processing: "+line, file=sys.stderr)
            vocab.update(Counter(line.strip().split()))
    return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()
