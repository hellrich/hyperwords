from collections import Counter
from math import sqrt
from random import Random

from docopt import docopt


def main():
    args = docopt("""
    Usage:
        ngram2pairs.py [options] <corpus>
    
    Options:
        --thr NUM    The minimal word count for being in the vocabulary [default: 100]
        --win NUM    Window size [default: 4]      
        --dyn        Dynamic context windows
        --sub NUM    Subsampling threshold [default: 0]
        --del        Delete out-of-vocabulary and subsampled placeholders
    """)
#--pos        Positional contexts
    corpus_file = args['<corpus>']
    thr = int(args['--thr'])
    win = int(args['--win'])
    #pos = args['--pos']
    dyn = args['--dyn']
    subsample = float(args['--sub'])
    sub = subsample != 0
    d3l = args['--del']

    vocab = read_vocab(corpus_file, thr)
    corpus_size = sum(vocab.values())

    subsample *= corpus_size
    subsampler = dict([(word, 1 - sqrt(subsample / count)) for word, count in vocab.items() if count > subsample])

    rnd = Random(17)
    counts = Counter()
    with open(corpus_file) as f:
        for line in f:
            text, year, match_count, volume_count = line.split("\t")
            tokens_outer = text.lower().split(" ")
            #change also belwo for normalized
            #if self.normalized == None:
          #          parts[self.TEXT] = parts[self.TEXT]
            #else:
            #        parts[self.TEXT] = [self.normalized[word].lower() if word in self.normalized else word.lower() for word in parts[self.TEXT].split(" ")]
            
            for i in range(int(match_count)):
                tokens = [t if t in vocab else None for t in tokens_outer]
                if sub:
                    tokens = [t if t not in subsampler or rnd.random() > subsampler[t] else None for t in tokens]
                if d3l:
                    tokens = [t for t in tokens if t is not None]

                len_tokens = len(tokens)

                for i, tok in enumerate(tokens):
                    if tok is not None:
                        if dyn:
                            dynamic_window = rnd.randint(1, win)
                        else:
                            dynamic_window = win
                        start = i - dynamic_window
                        if start < 0:
                            start = 0
                        end = i + dynamic_window + 1
                        if end > len_tokens:
                            end = len_tokens

                        #if pos:
                        #    output = '\n'.join([row for row in [tok + ' ' + tokens[j] + '_' + str(j - i) for j in xrange(start, end) if j != i and tokens[j] is not None] if len(row) > 0]).strip()
                        #else:
                        for pair in [(tok,tokens[j]) for j in xrange(start, end) if j != i and tokens[j] is not None]:
                            counts[pair] += 1
    for tup,c in counts.items():
        print(str(c)+" "+tup[0]+" "+tup[1])
        
def read_vocab(corpus_file, thr):
    vocab = Counter()
    with open(corpus_file) as f:
        for line in f:
            text, year, match_count, volume_count = line.split("\t")
            match_count = int(match_count)
            for token in text.lower().split(" "):
                vocab[token] += match_count
    return dict([(token, count) for token, count in vocab.items() if count >= thr])


if __name__ == '__main__':
    main()
