import os
import string
from collections import Counter
from docopt import docopt
import sys
import codecs

def main():
    args = docopt("""
    Usage:
        google_books_parts2counts.py [options] <path> <outpath> <start> <end>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 2]
        --step NUM   Time step between start and end [default: 10]
        --lowercase
    """)

    path = args['<path>']
    outpath = args['<outpath>']
    thr = int(args['--thr'])
    win = int(args['--win'])
    start = int(args['<start>'])
    end = int(args['<end>']) + 1
    step = int(args['--step'])  
    lowercase=args['--lowercase']
    if (end - start) % step != 0:
        raise Exception("Timespan not divisible by step!")
    for x in range(start, end - step + 1, step):
        print "Processing"+str(x)
        vocab = Counter()
        for y in range(x, x + step, 1):
            add_vocab(vocab,os.path.join(path, str(y)),lowercase)
        vocab = dict([(token, count) for token, count in vocab.items() if count >= thr])
        counts = Counter()
        for y in range(x, x + step, 1):
            add_counts(counts,os.path.join(path, str(y)),vocab,win,lowercase)
        outdir = os.path.join(outpath,str(x))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        store_counts(counts,os.path.join(outdir, "counts"))

def store_counts(counts, outfile):
    with codecs.open(outfile, "w", encoding="utf-8") as target_file:
        for pair, count in counts.items():
            target_file.write(str(count)+" "+pair[0] + " " + pair[1]+"\n")

def add_counts(counts,corpus_file,vocab,win,lowercase):
    with codecs.open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            text, year, match_count, volume_count = line.split("\t")
            if lowercase:
                text = text.lower()
            n = int(match_count)
            tokens = [t if t in vocab else None for t in text.strip().split()]
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
                        counts[pair] += n

def add_vocab(vocab, corpus_file, lowercase):
    with codecs.open(corpus_file) as f:
        for line in f:
            text, year, match_count, volume_count = line.split("\t")
            if lowercase:
                text = text.lower()
            n = int(match_count)
            for token in text.strip().split():
                if token not in string.punctuation:
                    vocab[token] += n

if __name__ == '__main__':
    main()
