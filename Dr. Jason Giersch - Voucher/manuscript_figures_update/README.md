# Manuscript figures update

Reproducible source for the figures/tables referenced by name in the draft
("How Americans Think about School Vouchers") that weren't previously
scripted in `../figures` or `../scripts`:

- `manuscript_figures.py` — generates `fig_manuscript_1_perceived_benefits.png`
  (Figure 1: sorted, single 100%-stacked bar, matching the manuscript caption
  "Sorted by Combined Benefit") and `fig_manuscript_2_beneficiary_coefficients.png`
  (Figure 2: forest plot of the six Table 3 regression models).
- `table4_framing_regression.py` — bivariate + controlled OLS models behind
  Table 4 (equity vs. specialty/efficiency framing effects); output in
  `table4_results.txt`.

Run either script from this folder with the raw
`POLSLab_Fall25_National_Labels_with_Background.xlsx` file present alongside
it (gitignored, not committed — see repo `.gitignore`).

**Sample size note:** both scripts apply only the attention-check filter
(column `22`). That gives n ≈ 982-992 per model, slightly above the
manuscript's reported n = 948-960. Coefficient signs, relative magnitudes,
and significance patterns match the manuscript closely, but an additional
QC exclusion (e.g. a response-duration/speeder cutoff) was evidently applied
in the original analysis and isn't identifiable from the label export alone.
Apply that same filter here before treating these as final replication
figures.
