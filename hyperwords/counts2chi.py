from docopt import docopt
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
import sys

from representations.matrix_serializer import save_matrix, save_vocabulary, load_count_vocabulary


def main():
    args = docopt("""
    Usage:
        counts2chi.py [options] <counts> <output_path>

    Options:
        --cds NUM    Context distribution smoothing [default: 1.0]
    """)

    counts_path = args['<counts>']
    vectors_path = args['<output_path>']
    cds = float(args['--cds'])

    counts, iw, ic = read_counts_matrix(counts_path)
    chi = calc_chi(counts, cds)

    save_matrix(vectors_path, chi)
    save_vocabulary(vectors_path + '.words.vocab', iw)
    save_vocabulary(vectors_path + '.contexts.vocab', ic)


def read_counts_matrix(counts_path):
    """
    Reads the counts into a sparse matrix (CSR) from the count-word-context textual format.
    """
    words = load_count_vocabulary(counts_path + '.words.vocab')
    contexts = load_count_vocabulary(counts_path + '.contexts.vocab')
    words = list(words.keys())
    contexts = list(contexts.keys())
    iw = sorted(words)
    ic = sorted(contexts)
    wi = dict([(w, i) for i, w in enumerate(iw)])
    ci = dict([(c, i) for i, c in enumerate(ic)])

    counts = csr_matrix((len(wi), len(ci)), dtype=np.float32)
    tmp_counts = dok_matrix((len(wi), len(ci)), dtype=np.float32)
    update_threshold = 100000
    i = 0
    with open(counts_path) as f:
        for line in f:
            count, word, context = line.strip().split()
            if word in wi and context in ci:
                tmp_counts[wi[word], ci[context]] = int(count)
            i += 1
            if i == update_threshold:
                counts = counts + tmp_counts.tolil()
                tmp_counts = dok_matrix((len(wi), len(ci)), dtype=np.float32)
                i = 0
    counts = counts + tmp_counts.tocsr()

    return counts, iw, ic


def calc_chi(counts, cds):
    """
    super stupid test implementation
    Counts = w1,w2 matrix with count for this combiantion, not a contingency table!
    """

    expected = get_expected(counts, cds)
    counts.data = np.square(counts.data - expected.data)
    counts.data /= expected.data

    return counts


def get_expected(counts, cds):
    sum_w = np.array(counts.sum(axis=1))[:, 0]  # horizontal, word
    sum_c = np.array(counts.sum(axis=0))[0, :]  # vertical, context

    if cds != 1:
        sum_c = sum_c ** cds

    # sparse
    expected = counts.copy()
    expected.data[expected.data != 0] = 1

    # real calculation
    expected = multiply_by_rows(expected, sum_w)
    expected = multiply_by_columns(expected, sum_c)
    expected.data /= sum_c.sum()  # performed better than sum_w.sum()
    # c proabably better as e(w,c) = p(w) * p(c) * N and p(w) = count(W) / N

    return expected


def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())

if __name__ == '__main__':
    main()
