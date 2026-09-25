#!/usr/bin/env python3
"""
Multiple-testing correction and power for the pairwise association matrix
==========================================================================
Reproduces the two statements in the Methods (Statistical analysis):

  1. Benjamini-Hochberg correction (q < 0.05) across all tested pairs, and how
     many associations with V >= 0.20 and V >= 0.40 are retained.
  2. The smallest Cramer's V detectable with 80% power at alpha = 0.05 for a
     binary pair at the median pairwise sample size (n = 304).

INPUT
  cramers_v_pairs_long.csv   (written by tmd_analysis.py), placed in the same
  folder as this script

USAGE
  python fdr_power.py
  python fdr_power.py --pairs path/to/file.csv   # to use a file elsewhere

Expected output for the manuscript version:
  BH retained: 133 of 331
  V>=0.20: 84 of 103
  V>=0.40: 25 of 26
  Detectable V at n=304: 0.161
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats, optimize


def benjamini_hochberg(p):
    """Return BH-adjusted q values in the original order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    q = np.empty(m)
    q[order] = q_sorted
    return q


def detectable_v(n, df=1, alpha=0.05, power=0.80):
    """Smallest effect size (Cramer's V = Cohen's w for df = 1) detectable
    with the given power, from the noncentral chi-squared distribution."""
    crit = stats.chi2.ppf(1 - alpha, df)
    return optimize.brentq(
        lambda w: 1 - stats.ncx2.cdf(crit, df, n * w ** 2) - power, 1e-4, 2)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--pairs', default=os.path.join(here, 'cramers_v_pairs_long.csv'))
    ap.add_argument('--n', type=int, default=304,
                    help='pairwise sample size for the power calculation')
    args = ap.parse_args()

    L = pd.read_csv(args.pairs).dropna(subset=['cramers_v', 'p_value'])
    L['q'] = benjamini_hochberg(L['p_value'])
    sig = L['q'] < 0.05

    print('BH retained:', int(sig.sum()), 'of', len(L))
    for t in (0.20, 0.40):
        strong = L['cramers_v'] >= t
        print(f'V>={t:.2f}:', int((sig & strong).sum()), 'of', int(strong.sum()))
    print(f'Detectable V at n={args.n}:', round(detectable_v(args.n), 3))


if __name__ == '__main__':
    main()
