"""

  Figure 1 — Distribution of Perceived Benefits of School Voucher Programs
             Across Population Groups (sorted by combined benefit; single
             100%-stacked horizontal bar, not six separate small multiples)

  Figure 2 — Estimated Regression Coefficients Predicting Perceived
             Beneficiaries of School Voucher Programs (forest / dot-whisker
             plot of the six OLS models behind Table 3)

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'POLSLab_Fall25_National_Labels_with_Background.xlsx')
df = pd.read_excel(DATA_PATH)

# ─────────────────────────────────────────────
# COLOR PALETTE (matches scripts/analysis.py)
# ─────────────────────────────────────────────
BLUE = '#2563EB'
TEAL = '#0D9488'
AMBER = '#D97706'
RED = '#DC2626'
GRAY = '#6B7280'
DARK = '#1F2937'

# ─────────────────────────────────────────────
# QUALITY CONTROL
# Drop failed attention check ("22" question). Replace/extend this with
# whatever duration or additional QC filter produced the manuscript's
# analytic n (953-960) if you want an exact match.
# ─────────────────────────────────────────────
df = df[df[22].astype(str).str.strip() == '22'].copy()

# ─────────────────────────────────────────────
# RECODE ORDINAL LABEL COLUMNS TO NUMERIC
# ─────────────────────────────────────────────
know_map = {'Not knowledgeable at all': 1, 'Somewhat knowledgable': 2, 'Very knowledgeable': 3}
support_map = {'Strongly oppose': 1, 'Somewhat oppose': 2, 'Not sure or neutral': 3,
               'Somewhat support': 4, 'Strongly support': 5}
benefit_map = {'They are greatly harmed': 1, 'They are slightly harmed': 2,
               'No effect or not sure': 3, 'They somewhat benefit': 4, 'They greatly benefit': 5}
ideo_map = {'Very liberal': 1, 'Liberal': 2, 'Somewhat liberal': 3, 'Moderate': 4,
            'Somewhat conservative': 5, 'Conservative': 6, 'Very conservative': 7}

df['VKnow_n'] = df['VKnow'].map(know_map)
df['Vsupport_n'] = df['Vsupport'].map(support_map)
df['Ideology_n'] = df['Ideology'].map(ideo_map)  # higher = more conservative ("Conservative" in Table 3)

benefit_cols = [
    ('VReligious', 'Religious\nFamilies'),
    ('VWealthy', 'Wealthy\nFamilies'),
    ('VUrban', 'Urban\nFamilies'),
    ('VPoor', 'Poor\nFamilies'),
    ('Vswd', 'Students with\nDisabilities'),
    ('VRural', 'Rural\nFamilies'),
]
for col, _ in benefit_cols:
    df[col + '_n'] = df[col].map(benefit_map)

# ═══════════════════════════════════════════════════════════
# FIGURE 1 — SORTED 100%-STACKED HORIZONTAL BAR
# Palette/style matches the version already in the manuscript (kept as-is
# per author preference): red / orange / gray / light-blue / dark-blue.
# ═══════════════════════════════════════════════════════════
effect_order = ['They are greatly harmed', 'They are slightly harmed',
                'No effect or not sure', 'They somewhat benefit', 'They greatly benefit']
seg_labels = ['Greatly Harmed', 'Slightly Harmed', 'No Effect / Not Sure', 'Somewhat Benefit', 'Greatly Benefit']
seg_colors = ['#C0392B', '#ED7D31', '#D9D9D9', '#9DC3E6', '#4472C4']
seg_text_colors = ['white', 'black', 'black', 'black', 'white']

rows = []
for col, label in benefit_cols:
    counts = df[col].value_counts()
    total = counts.sum()
    pcts = [counts.get(cat, 0) / total * 100 for cat in effect_order]
    combined_benefit = pcts[3] + pcts[4]  # somewhat + greatly benefit
    rows.append({'label': label.replace('\n', ' '), 'pcts': pcts, 'n': total, 'combined_benefit': combined_benefit})

# Sort by combined benefit, descending (matches manuscript caption)
rows.sort(key=lambda r: r['combined_benefit'], reverse=True)

fig1, ax = plt.subplots(figsize=(11, 5.5))
y_pos = np.arange(len(rows))[::-1]  # top row = highest combined benefit

left = np.zeros(len(rows))
for i, (seg_label, color) in enumerate(zip(seg_labels, seg_colors)):
    vals = np.array([r['pcts'][i] for r in rows])
    bars = ax.barh(y_pos, vals, left=left, color=color, edgecolor='white',
                    linewidth=0.8, height=0.62, zorder=3, label=seg_label)
    for y, v, l in zip(y_pos, vals, left):
        if v > 3:
            ax.text(l + v / 2, y, f'{v:.1f}%', ha='center', va='center',
                    fontsize=8.5, color=seg_text_colors[i])
    left += vals

ax.set_yticks(y_pos)
ax.set_yticklabels([r['label'] for r in rows], fontsize=10.5, color='black')
ax.set_xlabel('Percentage of Respondents (%)', fontsize=10, color='black')
ax.set_xlim(0, 100)
ax.set_title('Perceived Impact of School Voucher Programs (Sorted by Combined Benefit)',
             fontsize=12, color='black', pad=12)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(title='Response Category', loc='center left', bbox_to_anchor=(1.01, 0.5),
          fontsize=9, frameon=True, edgecolor='#CCCCCC')

fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, 'fig_manuscript_1_perceived_benefits.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print('Figure 1 (sorted stacked bar) saved.')

# ═══════════════════════════════════════════════════════════
# FIGURE 2 — REGRESSION COEFFICIENTS, TABLE 3 MODELS
# Redesigned as three small-multiple panels (one per predictor) with
# outcomes as directly-labeled, sorted rows instead of a single plot with
# an 18-way dodge and a color-matched legend — removes the need to
# cross-reference colors against a legend to read any one estimate.
# ═══════════════════════════════════════════════════════════
predictors = ['VKnow_n', 'Vsupport_n', 'Ideology_n']
predictor_titles = {'VKnow_n': 'Voucher Knowledge', 'Vsupport_n': 'Voucher Support', 'Ideology_n': 'Ideology (Conservative)'}

outcome_order = ['VPoor_n', 'VWealthy_n', 'VReligious_n', 'VUrban_n', 'VRural_n', 'Vswd_n']
outcome_labels = {'VPoor_n': 'Poor families', 'VWealthy_n': 'Wealthy families', 'VReligious_n': 'Religious families',
                   'VUrban_n': 'Urban families', 'VRural_n': 'Rural families', 'Vswd_n': 'Students with disabilities'}

results = {}
for oc in outcome_order:
    sub = df[[oc] + predictors].dropna()
    X = sm.add_constant(sub[predictors])
    y = sub[oc]
    model = sm.OLS(y, X).fit()
    ci = model.conf_int(alpha=0.05)
    results[oc] = {'coef': model.params, 'lo': ci[0], 'hi': ci[1], 'n': len(sub), 'r2': model.rsquared}
    print(f"{oc}: n={len(sub)}, R2={model.rsquared:.3f}, coefs={model.params.round(3).to_dict()}")

DOT = '#2563EB'
xmin = min(results[oc]['lo'][p] for oc in outcome_order for p in predictors) - 0.05
xmax = max(results[oc]['hi'][p] for oc in outcome_order for p in predictors) + 0.05

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)

for ax, pred in zip(axes2, predictors):
    # sort outcomes by coefficient size, largest (most positive) at top
    ordered = sorted(outcome_order, key=lambda oc: results[oc]['coef'][pred])
    ys = range(len(ordered))
    for y, oc in zip(ys, ordered):
        r = results[oc]
        coef, lo, hi = r['coef'][pred], r['lo'][pred], r['hi'][pred]
        sig = lo > 0 or hi < 0
        color = DOT if sig else '#9CA3AF'
        ax.plot([lo, hi], [y, y], color=color, linewidth=1.6, zorder=2, alpha=1 if sig else 0.6)
        ax.scatter([coef], [y], color=color, s=42, zorder=3, edgecolor='white', linewidth=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([outcome_labels[oc] for oc in ordered], fontsize=9.5, color='black')
    ax.axvline(0, color='black', linestyle='--', linewidth=1, zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_title(predictor_titles[pred], fontsize=11, fontweight='bold', color='black', pad=8)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

fig2.supxlabel('Unstandardized regression coefficient (95% CI)  •  gray = not significant at p<.05', fontsize=9, color='#4B5563')
fig2.suptitle('Figure 2. Estimated Regression Coefficients Predicting Perceived Beneficiaries\nof School Voucher Programs, by Predictor',
              fontsize=13, fontweight='bold', color='black', y=1.06)

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, 'fig_manuscript_2_beneficiary_coefficients.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print('Figure 2 (faceted coefficient plot) saved.')
