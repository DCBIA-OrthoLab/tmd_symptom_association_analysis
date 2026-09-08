# TMD symptom association analysis

Regenerates the results reported in the manuscript from the two raw inputs.

Analysis code and association matrices for a population-level study of
temporomandibular disorder symptom structure in 1,323 patients.

## What is here
- `tmd_analysis.py` — the full analysis
- `cramers_v_matrix.csv` — pairwise bias-corrected Cramer's V among clinical data elements
- `cramers_v_pairs_long.csv` — one row per pair, with V, complete-case n, and p value
- `cramers_v_sample_sizes.csv` — per-pair complete-case counts
- `requirements.txt` — package versions

## Inputs the script expects
Neither input file is included here. Both are patient-level and their release is
subject to institutional approval.

- `Table_ModelOutput_1401patients_46features.xlsx` — extraction output, one row per patient
- `pronoun_screen.csv` — pronoun counts used to derive patient sex

The clinical notes from which these were generated cannot be shared. The extraction
pipeline that produced them is described in the source publication and is a separate
artefact from this repository.

## Run
```
pip install pandas numpy scipy scikit-learn matplotlib openpyxl
python tmd_analysis.py \
    --data Table_ModelOutput_1401patients_46features.xlsx \
    --pronouns pronoun_screen.csv \
    --out output
```

Optional: `--exclude-treatment` drops appliance history, current appliance,
physical therapy status, pain relieving factors and current medications from the
matrix. These encode clinical decisions rather than patient symptoms. The
published matrix includes them; the flag reproduces the sensitivity analysis.

## Outputs
| file | contents |
|---|---|
| `cramers_v_matrix.csv` | full association matrix |
| `cramers_v_pairs_long.csv` | one row per pair: V, n, p |
| `cramers_v_sample_sizes.csv` | per-pair complete-case counts |
| `table1_neck_pain.csv` | Table 1 |
| `supplementary_S2_by_age.csv` | Supplementary Table S2 |
| `supplementary_S3_by_sex.csv` | Supplementary Table S3 |
| `sex_interaction_all_pairs.csv` | every pair tested for a sex interaction |
| `key_results.txt`, `key_results.json` | every statistic quoted in the text |

## Verified against the manuscript

Exact:
- 1,401 records, 78 excluded, cohort 1,323
- age documented 1,112 (84.0%), mean 30.7, SD 13.8, median 28, IQR 18–42
- sex resolved 1,322 of 1,323, 76.6% female
- 332 of 351 pairs estimable; median V 0.107; 232 pairs below 0.20; 26 at or above 0.40
- pair n from 30 to 957, median 304
- Table 1 values and sample sizes
- definitional overlap check: 0.741, n = 166
- sex interaction: vertigo × earache 15.19 vs 3.84 (p = 0.0008); tinnitus × earache
  16.79 vs 4.54 (p = 0.0031); tinnitus × vertigo 19.38 vs 5.78 (p = 0.0053)

Close but not identical, and the reason is documented:
- matched comparison gives 0.817 / 0.583 / 0.548 against 0.817 / 0.579 / 0.567 in the
  manuscript. The neck–muscle value and the ordering reproduce exactly; the two
  joint-involving values differ slightly because this script dichotomises TMJ pain
  rating at the sample median. **Set the threshold to whatever the original analysis
  used before depositing.**
- cophenetic correlation 0.799 against 0.798, from a marginally different
  well-covered feature subset.
- otologic within-group mean depends on whether hearing loss is included; the
  manuscript's 0.410 is the tinnitus/vertigo/earache trio.

## Citation
If you use this code or the association matrices, please cite the manuscript.
[Full citation and DOI to be added on acceptance.]

## License
MIT.
