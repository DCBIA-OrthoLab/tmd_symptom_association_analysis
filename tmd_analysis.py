#!/usr/bin/env python3
"""
TMD symptom association analysis
================================
Regenerates every number, table and figure reported in the manuscript from the
two raw inputs.

INPUTS
  Table_ModelOutput_1401patients_46features.xlsx   extraction output, one row per patient
  pronoun_screen.csv                               pronoun counts used to derive sex

OUTPUTS  (written to ./output)
  cramers_v_matrix.csv          full association matrix
  cramers_v_pairs_long.csv      one row per pair: V, n, p
  cramers_v_sample_sizes.csv    per-pair complete-case counts
  table1_neck_pain.csv          Table 1
  supplementary_S2_by_age.csv   Supplementary Table S2
  supplementary_S3_by_sex.csv   Supplementary Table S3
  key_results.txt               every statistic quoted in the text
  figure_*.png / .pdf           figures

USAGE
  python tmd_analysis.py --data <xlsx> --pronouns <csv> --out output

Python 3.12; pandas, numpy, scipy, scikit-learn, matplotlib, openpyxl.
"""

import argparse, itertools, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import squareform

RNG = np.random.default_rng(20260908)
N_BOOT = 4000
MIN_PAIR_N = 30          # pairs below this are not estimated
MIN_LEVELS, MAX_LEVELS = 2, 25
MIN_OBSERVED = 50        # a feature needs this many observed values
AGE_MIN, AGE_MAX = 12, 59
AGE_BINS, AGE_LABELS = [12, 20, 40, 60], ['12-19', '20-39', '40-59']

MISSING = {'unknown', 'nan', '', 'none', 'n/a'}


# ----------------------------------------------------------------------------
# loading and cohort construction
# ----------------------------------------------------------------------------
def load_cohort(data_path, pronoun_path):
    df = pd.read_excel(data_path)
    n_raw = len(df)

    age = pd.to_numeric(df['patient_age'], errors='coerce')
    # patients with no recorded age are RETAINED; only documented out-of-range excluded
    drop = (age < AGE_MIN) | (age > AGE_MAX)
    df = df[~drop].copy()
    df['age'] = pd.to_numeric(df['patient_age'], errors='coerce')

    pr = pd.read_csv(pronoun_path)
    df = df.merge(pr, on='patient_id', how='left')

    def sex_of(r):
        if r.get('category') == 'female_proxy':
            return 'F'
        if r.get('category') == 'male_proxy':
            return 'M'
        m, f = r.get('male_pronoun_count', 0), r.get('female_pronoun_count', 0)
        if pd.isna(m) or pd.isna(f):
            return None
        if m > f:
            return 'M'
        if f > m:
            return 'F'
        return None

    df['sex'] = df.apply(sex_of, axis=1)
    df['age_group'] = pd.cut(df['age'], AGE_BINS, labels=AGE_LABELS, right=False)

    meta = dict(
        records_available=int(n_raw),
        excluded_out_of_range=int(drop.sum()),
        cohort=int(len(df)),
        age_documented=int(df['age'].notna().sum()),
        age_missing=int(df['age'].isna().sum()),
        age_mean=float(df['age'].mean()),
        age_sd=float(df['age'].std()),
        age_median=float(df['age'].median()),
        age_q1=float(df['age'].quantile(.25)),
        age_q3=float(df['age'].quantile(.75)),
        sex_resolved=int(df['sex'].notna().sum()),
        pct_female=float(100 * (df['sex'] == 'F').sum() / df['sex'].notna().sum()),
    )
    return df, meta


def clean(series):
    """Map documented values to themselves and every missing marker to NaN."""
    v = series.astype(str).str.lower().str.strip()
    return v.where(~v.isin(MISSING) & series.notna(), np.nan)


def analysis_features(df, exclude_treatment=False):
    """Features eligible for the association matrix, per the Methods criteria."""
    drop = {'patient_id', 'age', 'sex', 'age_group', 'category',
            'male_pronoun_count', 'female_pronoun_count', 'patient_age'}
    # Treatment-history variables encode clinical decisions rather than patient
    # symptoms. The published matrix includes them; pass exclude_treatment=True
    # to reproduce the sensitivity analysis that leaves them out.
    if exclude_treatment:
        drop |= {'appliance_history', 'current_appliance', 'physical_therapy_status',
                 'pain_relieving_factors', 'current_medications'}
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        v = clean(df[c])
        k = v.nunique(dropna=True)
        if v.notna().sum() >= MIN_OBSERVED and MIN_LEVELS <= k <= MAX_LEVELS:
            keep.append(c)
    return sorted(keep)


# ----------------------------------------------------------------------------
# association statistics
# ----------------------------------------------------------------------------
def cramers_v(a, b, correct=True):
    """Bias-corrected Cramer's V (Bergsma 2013). Returns (V, n, p)."""
    a = pd.Series(np.asarray(a)); b = pd.Series(np.asarray(b))
    s = pd.DataFrame({'a': a, 'b': b}).dropna()
    n = len(s)
    if n < MIN_PAIR_N:
        return np.nan, n, np.nan
    t = pd.crosstab(s['a'], s['b'])
    if t.shape[0] < 2 or t.shape[1] < 2:
        return np.nan, n, np.nan
    chi2 = stats.chi2_contingency(t.values, correction=False)[0]
    p = stats.chi2_contingency(t.values)[1]
    r, k = t.shape
    phi2 = chi2 / n
    if correct:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    den = min(r - 1, k - 1)
    return (float(np.sqrt(phi2 / den)) if den > 0 else np.nan), n, float(p)


def association_matrix(df, feats):
    V = pd.DataFrame(np.nan, index=feats, columns=feats, dtype=float)
    N = pd.DataFrame(0, index=feats, columns=feats, dtype=int)
    rows = []
    C = {f: clean(df[f]) for f in feats}
    for a, b in itertools.combinations(feats, 2):
        v, n, p = cramers_v(C[a], C[b])
        V.loc[a, b] = V.loc[b, a] = v
        N.loc[a, b] = N.loc[b, a] = n
        rows.append(dict(feature_1=a, feature_2=b, cramers_v=v,
                         n_complete_cases=n, p_value=p,
                         note='' if n >= MIN_PAIR_N else f'below min-pair-n ({MIN_PAIR_N})'))
    for f in feats:
        V.loc[f, f] = 1.0
    return V, N, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# matched comparison with bootstrap
# ----------------------------------------------------------------------------
def to_binary(series, name):
    """Dichotomise for matched comparison: booleans as-is, scores at the median."""
    v = clean(series)
    lower = v.dropna().unique()
    if set(map(str, lower)) <= {'true', 'false'}:
        return v.map({'true': 1, 'false': 0})
    num = pd.to_numeric(v, errors='coerce')
    if num.notna().sum() > 0.8 * v.notna().sum():
        cut = 0  # any pain vs none, threshold >= 1
        return (num > cut).astype(float).where(num.notna())
    top = v.value_counts().idxmax()
    return (v != top).astype(float).where(v.notna())


def matched_comparison(df, a, b, c, n_boot=N_BOOT):
    """
    Compare V(a,b) against V(a,c) and V(b,c) on the patients with all three
    documented, everything dichotomised so the tables share dimensions.
    """
    X = pd.DataFrame({k: to_binary(df[k], k) for k in (a, b, c)}).dropna()
    n = len(X)
    obs = {p: cramers_v(X[p[0]], X[p[1]])[0] for p in [(a, b), (a, c), (b, c)]}
    idx = np.arange(n)
    diffs = {('ab', 'ac'): [], ('ab', 'bc'): []}
    for _ in range(n_boot):
        s = X.iloc[RNG.choice(idx, n, replace=True)].reset_index(drop=True)
        vab = cramers_v(s[a], s[b])[0]
        vac = cramers_v(s[a], s[c])[0]
        vbc = cramers_v(s[b], s[c])[0]
        diffs[('ab', 'ac')].append(vab - vac)
        diffs[('ab', 'bc')].append(vab - vbc)
    out = dict(n=n, V={f'{x}|{y}': obs[(x, y)] for x, y in obs})
    for key, d in diffs.items():
        d = np.array([x for x in d if np.isfinite(x)])
        out[f'diff_{key[0]}_minus_{key[1]}'] = dict(
            point=float(np.mean(d)), ci_low=float(np.percentile(d, 2.5)),
            ci_high=float(np.percentile(d, 97.5)),
            n_at_or_below_zero=int((d <= 0).sum()), n_boot=int(len(d)))
    return out


# ----------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------
def cluster_structure(V, N, min_n=100):
    ok = [f for f in V.index if (N.loc[f] >= min_n).sum() >= len(V) * 0.5]
    sub = V.loc[ok, ok].copy()
    sub = sub.fillna(sub.stack().median())
    D = 1 - sub.values
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    cond = squareform(D, checks=False)
    Z = linkage(cond, method='average')
    coph = float(cophenet(Z, cond)[0])
    memberships = {int(k): dict(zip(ok, map(int, fcluster(Z, k, criterion='maxclust'))))
                   for k in range(2, 7)}
    return dict(features=ok, cophenetic=coph, memberships=memberships)


# ----------------------------------------------------------------------------
# stratified analyses
# ----------------------------------------------------------------------------
def odds_ratio(sub, a, b):
    s = pd.DataFrame({'a': to_binary(sub[a], a), 'b': to_binary(sub[b], b)}).dropna()
    t = pd.crosstab(s['a'], s['b'])
    if t.shape != (2, 2) or t.values.min() < 5:
        return None
    w, x, y, z = t.iloc[1, 1], t.iloc[1, 0], t.iloc[0, 1], t.iloc[0, 0]
    lor = np.log((w * z) / (x * y))
    se = np.sqrt(1 / w + 1 / x + 1 / y + 1 / z)
    return dict(logor=float(lor), se=float(se), or_=float(np.exp(lor)),
                n=int(len(s)), p=float(stats.fisher_exact(t.values)[1]))


def interaction_by_sex(df, pairs):
    out = []
    for a, b in pairs:
        f = odds_ratio(df[df.sex == 'F'], a, b)
        m = odds_ratio(df[df.sex == 'M'], a, b)
        if not f or not m:
            out.append(dict(pair=f'{a} x {b}', note='cells too small'))
            continue
        z = (f['logor'] - m['logor']) / np.sqrt(f['se'] ** 2 + m['se'] ** 2)
        out.append(dict(pair=f'{a} x {b}', or_female=f['or_'], n_female=f['n'],
                        or_male=m['or_'], n_male=m['n'],
                        interaction_p=float(2 * (1 - stats.norm.cdf(abs(z))))))
    return pd.DataFrame(out)


def prevalence_table(df, split, groups, colnames, feats_bin, feats_ord, feats_cat):
    rows = []
    for c in feats_bin:
        r = {'feature': c, 'measure': '% present (n documented)'}
        cols = []
        for g, cn in zip(groups, colnames):
            s = to_binary(df[df[split] == g][c], c).dropna()
            r[cn] = f'{100 * s.mean():.1f}% ({len(s)})'
            cols.append(s)
        s = df[[c, split]].copy(); s[c] = to_binary(s[c], c)
        t = pd.crosstab(s[split], s[c].dropna())
        r['p'] = float(stats.chi2_contingency(t.values)[1]) if t.shape[0] > 2 \
            else float(stats.fisher_exact(t.values)[1])
        rows.append(r)
    for c in feats_ord:
        r = {'feature': c, 'measure': 'median [IQR] (n)'}
        gs = []
        for g, cn in zip(groups, colnames):
            s = pd.to_numeric(clean(df[df[split] == g][c]), errors='coerce').dropna()
            r[cn] = f'{np.median(s):.0f} [{np.percentile(s,25):.0f}-{np.percentile(s,75):.0f}] ({len(s)})'
            gs.append(s.values)
        r['p'] = float(stats.kruskal(*gs)[1]) if len(gs) > 2 else float(stats.mannwhitneyu(*gs)[1])
        rows.append(r)
    for c in feats_cat:
        r = {'feature': c, 'measure': '% documented (n in group)'}
        for g, cn in zip(groups, colnames):
            sub = df[df[split] == g]
            r[cn] = f'{100 * clean(sub[c]).notna().mean():.1f}% ({len(sub)})'
        t = pd.crosstab(df[split], clean(df[c]).notna())
        r['p'] = float(stats.chi2_contingency(t.values)[1]) if t.shape[0] > 2 \
            else float(stats.fisher_exact(t.values)[1])
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--pronouns', required=True)
    ap.add_argument('--out', default='output')
    ap.add_argument('--exclude-treatment', action='store_true',
                    help='drop treatment-history variables from the matrix')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    O = lambda f: os.path.join(args.out, f)

    df, meta = load_cohort(args.data, args.pronouns)
    log = []
    P = lambda s: (print(s), log.append(s))

    P('=' * 72)
    P('COHORT')
    P('=' * 72)
    for k, v in meta.items():
        P(f'  {k:24} {v:.1f}' if isinstance(v, float) else f'  {k:24} {v}')

    feats = analysis_features(df, exclude_treatment=args.exclude_treatment)
    P(f'\n  features entering the matrix: {len(feats)}')

    V, N, long = association_matrix(df, feats)
    V.to_csv(O('cramers_v_matrix.csv'))
    N.to_csv(O('cramers_v_sample_sizes.csv'))
    long.to_csv(O('cramers_v_pairs_long.csv'), index=False)

    est = long[long['note'] == '']
    P(f'  estimable pairs: {len(est)} of {len(long)}')
    P(f'  median V {est.cramers_v.median():.3f} | pairs < 0.20: {(est.cramers_v < .2).sum()}'
      f' | pairs >= 0.40: {(est.cramers_v >= .4).sum()}')
    P(f'  pair n: min {est.n_complete_cases.min()}, median {int(est.n_complete_cases.median())},'
      f' max {est.n_complete_cases.max()}')

    # ---- Table 1
    P('\n' + '=' * 72); P('TABLE 1  neck pain associations'); P('=' * 72)
    t1 = est[(est.feature_1 == 'neck_pain_present') | (est.feature_2 == 'neck_pain_present')].copy()
    t1['partner'] = t1.apply(lambda r: r.feature_2 if r.feature_1 == 'neck_pain_present'
                             else r.feature_1, axis=1)
    t1 = t1.sort_values('cramers_v', ascending=False)[['partner', 'cramers_v', 'n_complete_cases']]
    t1.to_csv(O('table1_neck_pain.csv'), index=False)
    for _, r in t1.head(10).iterrows():
        P(f'  {r.cramers_v:.3f}  n={int(r.n_complete_cases):4d}  {r.partner}')

    # ---- matched comparison
    P('\n' + '=' * 72); P('MATCHED COMPARISON  (dichotomised, shared patients)'); P('=' * 72)
    mc = matched_comparison(df, 'neck_pain_present', 'muscle_symptoms_present', 'tmj_pain_rating')
    P(f'  n = {mc["n"]}')
    for k, v in mc['V'].items():
        P(f'  V {k:52} {v:.3f}')
    for k in [x for x in mc if x.startswith('diff_')]:
        d = mc[k]
        P(f'  {k}: {d["point"]:.3f}  95% CI [{d["ci_low"]:.3f}, {d["ci_high"]:.3f}]'
          f'  resamples <= 0: {d["n_at_or_below_zero"]}/{d["n_boot"]}')

    # ---- definitional overlap
    P('\n' + '=' * 72); P('DEFINITIONAL OVERLAP CHECK'); P('=' * 72)
    loc = clean(df['muscle_pain_location']).fillna('')
    excl = loc.str.contains('neck')
    v_all = cramers_v(clean(df['neck_pain_present']), clean(df['muscle_symptoms_present']))
    v_ex = cramers_v(clean(df.loc[~excl, 'neck_pain_present']),
                     clean(df.loc[~excl, 'muscle_symptoms_present']))
    P(f'  all patients:                V={v_all[0]:.3f}  n={v_all[1]}')
    P(f'  excluding neck-as-muscle-site ({int(excl.sum())} patients): V={v_ex[0]:.3f}  n={v_ex[1]}')

    # ---- clustering
    P('\n' + '=' * 72); P('HIERARCHICAL CLUSTERING'); P('=' * 72)
    cl = cluster_structure(V, N)
    P(f'  features: {len(cl["features"])}   cophenetic correlation: {cl["cophenetic"]:.3f}')
    oto = ['tinnitus_present', 'vertigo_present', 'earache_present', 'hearing_loss_present']
    for k, mem in cl['memberships'].items():
        got = {f: mem[f] for f in oto if f in mem}
        alone = len(set(got.values())) == 1 and \
            not any(mem[f] in got.values() for f in mem if f not in got)
        P(f'  k={k}: otologic memberships {got}  -> separate cluster: {alone}')

    # ---- otologic coupling
    P('\n' + '=' * 72); P('OTOLOGIC COUPLING'); P('=' * 72)
    within, across = [], []
    others = [f for f in feats if f not in oto]
    for a, b in itertools.combinations([f for f in oto if f in feats], 2):
        v = V.loc[a, b]
        if np.isfinite(v):
            within.append(v)
    for a in oto:
        for b in others:
            if a in V.index and np.isfinite(V.loc[a, b]):
                across.append(V.loc[a, b])
    P(f'  mean within otologic  {np.mean(within):.3f}  ({len(within)} pairs)')
    P(f'  mean otologic to rest {np.mean(across):.3f}  ({len(across)} pairs)')

    # ---- sex interaction
    P('\n' + '=' * 72); P('SEX INTERACTION IN COUPLING'); P('=' * 72)
    bin_feats = [f for f in feats if set(map(str, clean(df[f]).dropna().unique())) <= {'true', 'false'}]
    pairs = list(itertools.combinations(bin_feats, 2))
    inter = interaction_by_sex(df, pairs)
    inter.to_csv(O('sex_interaction_all_pairs.csv'), index=False)
    tested = inter.dropna(subset=['interaction_p']).sort_values('interaction_p')
    P(f'  {len(tested)} pairs testable; Bonferroni threshold p < {0.05/len(tested):.4f}')
    for _, r in tested.head(6).iterrows():
        P(f'  {r.pair:56} OR_F={r.or_female:6.2f} (n={int(r.n_female):3d})'
          f'  OR_M={r.or_male:6.2f} (n={int(r.n_male):3d})  p={r.interaction_p:.4f}')

    # ---- supplementary tables
    BIN = [f for f in ['neck_pain_present', 'muscle_symptoms_present', 'tinnitus_present',
                       'vertigo_present', 'earache_present', 'hearing_loss_present',
                       'sleep_apnea_diagnosed', 'airway_obstruction_present'] if f in df.columns]
    ORD = [f for f in ['tmj_pain_rating', 'headache_intensity', 'disability_rating', 'diet_score',
                       'jaw_function_score', 'average_daily_pain_intensity'] if f in df.columns]
    CAT = [f for f in ['jaw_clicking', 'jaw_crepitus', 'muscle_pain_score',
                       'headache_frequency'] if f in df.columns]
    s2 = prevalence_table(df[df.age_group.notna()], 'age_group', AGE_LABELS,
                          ['12-19 years', '20-39 years', '40-59 years'], BIN, ORD, CAT)
    s3 = prevalence_table(df[df.sex.notna()], 'sex', ['F', 'M'],
                          ['Female', 'Male'], BIN, ORD, CAT)
    s2.to_csv(O('supplementary_S2_by_age.csv'), index=False)
    s3.to_csv(O('supplementary_S3_by_sex.csv'), index=False)
    P('\n  wrote supplementary tables S2 (age) and S3 (sex)')


    # ---- age-stratified associations
    P('\n' + '=' * 72); P('AGE-STRATIFIED ASSOCIATIONS'); P('=' * 72)
    df['age_group'] = pd.cut(df['age'], [12, 20, 40, 60], labels=['12-19', '20-39', '40-59'], right=False)
    age_feats = feats
    age_rows = []
    for a, b in itertools.combinations(age_feats, 2):
        row = dict(feature_1=a, feature_2=b)
        all_ok = True
        for g in ['12-19', '20-39', '40-59']:
            sub = df[df['age_group'] == g]
            ca = clean(sub[a])
            cb = clean(sub[b])
            v, n, p = cramers_v(ca, cb)
            row[f'V_{g}'] = v
            row[f'n_{g}'] = n
            if n < 40 or np.isnan(v):
                all_ok = False
        row['estimable_all_groups'] = all_ok
        age_rows.append(row)
    age_df = pd.DataFrame(age_rows)
    age_df.to_csv(O('age_stratified_pairs.csv'), index=False)
    est_age = age_df[age_df['estimable_all_groups']]
    P(f'  pairs estimable in all three groups (n>=40): {len(est_age)} of {len(age_df)}')
    if len(est_age) > 30:
        from scipy.stats import pearsonr as _pr
        r12 = _pr(est_age['V_12-19'], est_age['V_20-39'])[0]
        r23 = _pr(est_age['V_20-39'], est_age['V_40-59'])[0]
        r13 = _pr(est_age['V_12-19'], est_age['V_40-59'])[0]
        P(f'  Pearson r  12-19 vs 20-39: {r12:.3f}')
        P(f'  Pearson r  20-39 vs 40-59: {r23:.3f}')
        P(f'  Pearson r  12-19 vs 40-59: {r13:.3f}')

    # ---- split-half reliability ceiling
    P('\n  SPLIT-HALF CEILING (within single age group):')
    _rng = np.random.default_rng(11)
    ceilings = []
    for g in ['12-19', '20-39', '40-59']:
        sub_df = df[df['age_group'] == g].reset_index(drop=True)
        rs = []
        for _rep in range(20):
            idx = _rng.permutation(len(sub_df))
            h1 = sub_df.iloc[idx[:len(sub_df)//2]]
            h2 = sub_df.iloc[idx[len(sub_df)//2:]]
            v1, v2 = [], []
            for a, b in itertools.combinations(age_feats, 2):
                x = cramers_v(clean(h1[a]), clean(h1[b]))
                y = cramers_v(clean(h2[a]), clean(h2[b]))
                if not np.isnan(x[0]) and not np.isnan(y[0]) and x[1] >= 20 and y[1] >= 20:
                    v1.append(x[0]); v2.append(y[0])
            if len(v1) > 30:
                rs.append(_pr(v1, v2)[0])
        if rs:
            ceilings.append(np.mean(rs))
            P(f'    {g}: split-half r = {np.mean(rs):.3f} ({len(rs)} splits)')
    if ceilings:
        P(f'    mean ceiling: {np.mean(ceilings):.3f}')

    # ---- within-domain vs cross-domain
    P('\n' + '=' * 72); P('WITHIN-DOMAIN VS CROSS-DOMAIN'); P('=' * 72)
    DOMAIN_MAP = {
        'neck_pain_present': 'musculoskeletal', 'muscle_symptoms_present': 'musculoskeletal',
        'muscle_pain_score': 'musculoskeletal', 'muscle_pain_location': 'musculoskeletal',
        'tmj_pain_rating': 'pain', 'headache_intensity': 'pain',
        'average_daily_pain_intensity': 'pain', 'headache_frequency': 'pain',
        'headache_location': 'pain', 'pain_aggravating_factors': 'pain', 'joint_pain_areas': 'pain',
        'disability_rating': 'functional', 'diet_score': 'functional',
        'jaw_function_score': 'functional', 'jaw_clicking': 'functional',
        'jaw_crepitus': 'functional', 'maximum_opening': 'functional',
        'maximum_opening_without_pain': 'functional', 'joint_arthritis_location': 'functional',
        'disc_displacement': 'functional', 'jaw_locking': 'functional',
        'earache_present': 'otologic', 'tinnitus_present': 'otologic',
        'vertigo_present': 'otologic', 'hearing_loss_present': 'otologic',
        'sleep_disorder_type': 'sleep_airway', 'sleep_apnea_diagnosed': 'sleep_airway',
        'airway_obstruction_present': 'sleep_airway',
        'depression_present': 'systemic', 'fibromyalgia_present': 'systemic',
        'autoimmune_condition': 'systemic', 'migraine_history': 'systemic',
        'back_pain_present': 'systemic',
        'appliance_history': 'treatment', 'current_appliance': 'treatment',
        'physical_therapy_status': 'treatment', 'current_medications': 'treatment',
        'previous_medications': 'treatment', 'pain_relieving_factors': 'treatment',
        'adverse_reactions': 'treatment',
    }
    Vd = {}
    for _, r in long.iterrows():
        if r['note'] == '' or (isinstance(r['note'], float) and np.isnan(r['note'])):
            Vd[frozenset((r['feature_1'], r['feature_2']))] = r['cramers_v']
    all_in_matrix = sorted(set(long['feature_1']) | set(long['feature_2']))
    for dom in ['musculoskeletal', 'pain', 'functional', 'otologic', 'sleep_airway', 'systemic', 'treatment']:
        members = [f for f in all_in_matrix if DOMAIN_MAP.get(f) == dom]
        wi = [Vd[frozenset(p)] for p in itertools.combinations(members, 2) if frozenset(p) in Vd]
        cr = [Vd[frozenset((a, b))] for a in members for b in all_in_matrix
              if DOMAIN_MAP.get(b, 'x') != dom and frozenset((a, b)) in Vd]
        wi_clean = [x for x in wi if np.isfinite(x)]
        cr_clean = [x for x in cr if np.isfinite(x)]
        wi_str = f'{np.mean(wi_clean):.3f}' if wi_clean else 'n/a'
        cr_str = f'{np.mean(cr_clean):.3f}' if cr_clean else 'n/a'
        P(f'  {dom:20} within {wi_str:>6} ({len(wi_clean)} pairs)  cross {cr_str:>6} ({len(cr_clean)} pairs)')

    with open(O('key_results.txt'), 'w') as fh:
        fh.write('\n'.join(log) + '\n')
    json.dump({'cohort': meta, 'matched_comparison': mc,
               'clustering': {k: v for k, v in cl.items() if k != 'memberships'}},
              open(O('key_results.json'), 'w'), indent=2)
    P(f'\nAll outputs written to {args.out}/')


if __name__ == '__main__':
    main()
