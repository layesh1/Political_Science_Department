"""
Formalizes Table 4 of the manuscript ("Effects of Equity Framing on Support
for Voucher Program Restrictions rather than Efficiency Framing") as a
reproducible regression script, so the numbers behind fig4_experiment_means.png
(mean comparison / t-tests) have a matching regression-table source.

For each of the three regulatory scenarios (tuition barrier, religious
barrier, ability barrier), fits two OLS models on the combined
specialty/equity sample:
  Model 1 (bivariate):   eligibility ~ Treatment
  Model 2 (with controls): eligibility ~ Treatment + VKnow + Vsupport + Ideology

Outcome coding: 1 = definitely eligible ... 5 = definitely not eligible
(higher = more support for restricting/excluding the school).
Treatment: 1 = equity frame, 0 = specialty/efficiency frame.

See the note in manuscript_figures.py re: sample size / QC caveat — the same
applies here (n will be slightly larger than the manuscript's reported
n=948-956 because only the attention-check filter is applied).
"""

import os
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'POLSLab_Fall25_National_Labels_with_Background.xlsx')
df = pd.read_excel(DATA_PATH)

# Quality control: drop failed attention check (see manuscript_figures.py note)
df = df[df[22].astype(str).str.strip() == '22'].copy()

# ─────────────────────────────────────────────
# RECODE
# ─────────────────────────────────────────────
know_map = {'Not knowledgeable at all': 1, 'Somewhat knowledgable': 2, 'Very knowledgeable': 3}
support_map = {'Strongly oppose': 1, 'Somewhat oppose': 2, 'Not sure or neutral': 3,
               'Somewhat support': 4, 'Strongly support': 5}
ideo_map = {'Very liberal': 1, 'Liberal': 2, 'Somewhat liberal': 3, 'Moderate': 4,
            'Somewhat conservative': 5, 'Conservative': 6, 'Very conservative': 7}
eligibility_num = {
    'Yes, the school should definitely be eligible for vouchers': 1,
    'It probably should be eligible': 2,
    'Neutral or not sure': 3,
    'It probably should not be eligible': 4,
    'No, it definitely should not be eligible for vouchers': 5,
}

df['VKnow_n'] = df['VKnow'].map(know_map)
df['Vsupport_n'] = df['Vsupport'].map(support_map)
df['Ideology_n'] = df['Ideology'].map(ideo_map)

for col in ['TuitionRule', 'ReligionRule', 'SWDrule', 'TuitionRuleE', 'ReligionRuleE', 'SWDruleE']:
    df[col + '_num'] = df[col].map(eligibility_num)

scenarios = [
    ('TuitionRule_num', 'TuitionRuleE_num', 'Tuition barriers'),
    ('ReligionRule_num', 'ReligionRuleE_num', 'Religious barriers'),
    ('SWDrule_num', 'SWDruleE_num', 'Ability barriers'),
]

lines = []
lines.append('=' * 70)
lines.append('TABLE 4 REPLICATION — Effects of Equity Framing on Support for')
lines.append('Voucher Program Restrictions (vs. Efficiency/Specialty Framing)')
lines.append('=' * 70)

for spec_col, eq_col, label in scenarios:
    spec = df[[spec_col, 'VKnow_n', 'Vsupport_n', 'Ideology_n']].rename(columns={spec_col: 'y'})
    spec['Treatment'] = 0
    eq = df[[eq_col, 'VKnow_n', 'Vsupport_n', 'Ideology_n']].rename(columns={eq_col: 'y'})
    eq['Treatment'] = 1
    combined = pd.concat([spec, eq], ignore_index=True).dropna(subset=['y'])

    lines.append(f'\n--- {label} ---')

    # Model 1: bivariate
    m1_data = combined.dropna(subset=['Treatment'])
    X1 = sm.add_constant(m1_data[['Treatment']])
    m1 = sm.OLS(m1_data['y'], X1).fit()
    lines.append(f'Model 1 (bivariate): n = {len(m1_data)}, R2 = {m1.rsquared:.3f}')
    lines.append(f'  Treatment = {m1.params["Treatment"]:.3f}  (p = {m1.pvalues["Treatment"]:.4f})')

    # Model 2: with controls
    m2_data = combined.dropna(subset=['Treatment', 'VKnow_n', 'Vsupport_n', 'Ideology_n'])
    X2 = sm.add_constant(m2_data[['Treatment', 'VKnow_n', 'Vsupport_n', 'Ideology_n']])
    m2 = sm.OLS(m2_data['y'], X2).fit()
    lines.append(f'Model 2 (with controls): n = {len(m2_data)}, R2 = {m2.rsquared:.3f}')
    for var in ['Treatment', 'VKnow_n', 'Vsupport_n', 'Ideology_n']:
        lines.append(f'  {var:<12} = {m2.params[var]:+.3f}  (p = {m2.pvalues[var]:.4f})')

summary_text = '\n'.join(lines)
print(summary_text)

with open(os.path.join(script_dir, 'table4_results.txt'), 'w') as f:
    f.write(summary_text)

print('\nSaved table4_results.txt')
