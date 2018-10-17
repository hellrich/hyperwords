from collections import Counter
from math import sqrt

from docopt import docopt


def main():
    args = docopt("""
    Usage:
        corpus2counts.py [options] <corpus>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
    """)

    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])

    vocab = read_vocab(corpus_file, thr)
    corpus_size = sum(vocab.values())

    counts = Counter()
    with open(corpus_file) as f:
        for line in f:
            tokens = [t if t in vocab else None for t in line.strip().split()]
            len_tokens = len(tokens)
            for i, tok in enumerate(tokens):
                if tok is not None:
                    start = i - win
                    if start < 0:
                        start = 0
                    end = i + win + 1
                    if end > len_tokens:
                        end = len_tokens
                    for pair in [(tok, tokens[j]) for j in xrange(start, end) if j != i and tokens[j] is not None]:
                        counts[pair] += 1
    for pair, count in counts.items():
        print(str(count)+" "+pair[0] + " " + pair[1])


def read_vocab(corpus_file, thr):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            vocab.update(Counter(line.strip().split()))
    return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()
