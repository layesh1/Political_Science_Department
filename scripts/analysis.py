import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(os.path.join(script_dir, 'POLSLab_Fall25_National_Labels_with_Background.xlsx'))

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
BLUE = '#2563EB'
TEAL = '#0D9488'
AMBER = '#D97706'
RED = '#DC2626'
GRAY = '#6B7280'
LIGHT = '#F3F4F6'
DARK = '#1F2937'

# ─────────────────────────────────────────────
# ORDERED CATEGORY MAPS
# ─────────────────────────────────────────────
vin_order = ['Yes', 'No', 'Not sure']
vknow_order = ['Not knowledgeable at all', 'Somewhat knowledgable', 'Very knowledgeable']
vsupport_order = ['Strongly oppose', 'Somewhat oppose', 'Not sure or neutral', 'Somewhat support', 'Strongly support']
effect_order = ['They are greatly harmed', 'They are slightly harmed', 'No effect or not sure', 'They somewhat benefit', 'They greatly benefit']
eligibility_order = ['Yes, the school should definitely be eligible for vouchers',
                     'It probably should be eligible',
                     'Neutral or not sure',
                     'It probably should not be eligible',
                     'No, it definitely should not be eligible for vouchers']
eligibility_short = ['Definitely\neligible', 'Probably\neligible', 'Neutral /\nnot sure', 'Probably\nnot eligible', 'Definitely\nnot eligible']

effect_short = ['Greatly\nharmed', 'Slightly\nharmed', 'No effect /\nnot sure', 'Somewhat\nbenefit', 'Greatly\nbenefit']
support_short = ['Strongly\noppose', 'Somewhat\noppose', 'Not sure /\nneutral', 'Somewhat\nsupport', 'Strongly\nsupport']

# ─────────────────────────────────────────────
# HELPER: PERCENT BAR
# ─────────────────────────────────────────────
def pct_series(series, order):
    counts = series.value_counts()
    total = counts.sum()
    return pd.Series({k: counts.get(k, 0) / total * 100 for k in order})

def plot_dist(ax, series, order, short_labels, colors, title, n):
    pcts = pct_series(series.dropna(), order)
    bars = ax.bar(range(len(order)), pcts.values, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(short_labels, fontsize=8.5, color=DARK)
    ax.set_ylabel('Percent (%)', fontsize=9, color=GRAY)
    ax.set_title(title, fontsize=11, fontweight='bold', color=DARK, pad=8)
    ax.set_ylim(0, max(pcts.values) * 1.22)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.98, 0.97, f'n = {n}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=GRAY)
    for bar, val in zip(bars, pcts.values):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5, color=DARK)

# ─────────────────────────────────────────────
# FIGURE 1: BLOCK 1 — AWARENESS & SUPPORT
# ─────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('Block 1: Awareness & Support for School Vouchers', fontsize=14, fontweight='bold', color=DARK, y=1.01)

# VinState
colors_vin = [TEAL, RED, GRAY]
plot_dist(axes[0], df['VinState'], vin_order, ['Yes', 'No', 'Not sure'],
          colors_vin, 'Does your state offer a voucher program?\n(VinState)', df['VinState'].notna().sum())

# VKnow
colors_know = ['#93C5FD', '#3B82F6', '#1D4ED8']
plot_dist(axes[1], df['VKnow'], vknow_order,
          ['Not knowledgeable\nat all', 'Somewhat\nknowledgeable', 'Very\nknowledgeable'],
          colors_know, 'Self-reported knowledge about vouchers\n(VKnow)', df['VKnow'].notna().sum())

# Vsupport
colors_sup = [RED, '#FCA5A5', GRAY, '#86EFAC', TEAL]
plot_dist(axes[2], df['Vsupport'], vsupport_order, support_short,
          colors_sup, 'General support for voucher policies\n(Vsupport)', df['Vsupport'].notna().sum())

fig1.tight_layout()
fig1.savefig(os.path.join(script_dir, 'fig1_block1_awareness.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ─────────────────────────────────────────────
# FIGURE 2: BLOCK 2 — PERCEIVED EFFECTS
# ─────────────────────────────────────────────
effect_vars = [
    ('VPoor', 'Effects on\nPoor Families'),
    ('VWealthy', 'Effects on\nWealthy Families'),
    ('VReligious', 'Effects on\nReligious Families'),
    ('VUrban', 'Effects on\nUrban Families'),
    ('VRural', 'Effects on\nRural Families'),
    ('Vswd', 'Effects on Students\nwith Disabilities'),
]
colors_effect = [RED, '#FCA5A5', GRAY, '#86EFAC', TEAL]

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
fig2.suptitle('Block 2: Perceived Effects of Vouchers on Different Populations', fontsize=14, fontweight='bold', color=DARK, y=1.01)

for ax, (col, title) in zip(axes2.flatten(), effect_vars):
    plot_dist(ax, df[col], effect_order, effect_short, colors_effect,
              title, df[col].notna().sum())

fig2.tight_layout()
fig2.savefig(os.path.join(script_dir, 'fig2_block2_effects.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ─────────────────────────────────────────────
# BLOCK 3 & 4: FRAMING EXPERIMENT
# Specialty: TuitionRule, ReligionRule, SWDrule
# Equity:    TuitionRuleE, ReligionRuleE, SWDruleE
# ─────────────────────────────────────────────

# Build numeric coding for experiment vars (1=definitely eligible → 5=definitely not eligible)
# Lower = more supportive of eligibility
eligibility_num = {
    'Yes, the school should definitely be eligible for vouchers': 1,
    'It probably should be eligible': 2,
    'Neutral or not sure': 3,
    'It probably should not be eligible': 4,
    'No, it definitely should not be eligible for vouchers': 5,
}

for col in ['TuitionRule','ReligionRule','SWDrule','TuitionRuleE','ReligionRuleE','SWDruleE']:
    df[col + '_num'] = df[col].map(eligibility_num)

# ─────────────────────────────────────────────
# FIGURE 3: SIDE-BY-SIDE DISTRIBUTIONS
# ─────────────────────────────────────────────
exp_pairs = [
    ('TuitionRule', 'TuitionRuleE', 'Tuition School'),
    ('ReligionRule', 'ReligionRuleE', 'Religious School'),
    ('SWDrule', 'SWDruleE', 'Selective Cognitive School'),
]

colors_spec = ['#34D399', '#6EE7B7', GRAY, '#FCA5A5', RED]   # green→neutral→red (eligible→not)
colors_eq   = ['#60A5FA', '#93C5FD', GRAY, '#FCA5A5', RED]

fig3, axes3 = plt.subplots(3, 2, figsize=(14, 13))
fig3.suptitle('Blocks 3 & 4: Framing Experiment — Specialty vs. Equity Frame\nSchool Eligibility for Vouchers',
              fontsize=13, fontweight='bold', color=DARK, y=1.01)

for row, (spec_col, eq_col, label) in enumerate(exp_pairs):
    # Specialty
    ax_s = axes3[row, 0]
    n_s = df[spec_col].notna().sum()
    plot_dist(ax_s, df[spec_col], eligibility_order, eligibility_short,
              colors_spec, f'SPECIALTY frame\n{label}', n_s)
    ax_s.set_facecolor('#F0FDF4')

    # Equity
    ax_e = axes3[row, 1]
    n_e = df[eq_col].notna().sum()
    plot_dist(ax_e, df[eq_col], eligibility_order, eligibility_short,
              colors_eq, f'EQUITY frame\n{label}', n_e)
    ax_e.set_facecolor('#EFF6FF')

fig3.tight_layout()
fig3.savefig(os.path.join(script_dir, 'fig3_experiment_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ─────────────────────────────────────────────
# FIGURE 4: MEAN COMPARISON + T-TESTS
# ─────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(13, 5.5))
fig4.suptitle('Framing Experiment: Mean Eligibility Scores by Frame\n(1 = Definitely Eligible → 5 = Definitely Not Eligible)',
              fontsize=12, fontweight='bold', color=DARK, y=1.01)

for ax, (spec_col, eq_col, label) in zip(axes4, exp_pairs):
    spec_data = df[spec_col + '_num'].dropna()
    eq_data   = df[eq_col + '_num'].dropna()

    m_s = spec_data.mean()
    m_e = eq_data.mean()
    se_s = spec_data.sem()
    se_e = eq_data.sem()

    t_stat, p_val = stats.ttest_ind(spec_data, eq_data)

    bars = ax.bar(['Specialty\nFrame', 'Equity\nFrame'], [m_s, m_e],
                  color=['#34D399', '#60A5FA'], edgecolor='white', linewidth=0.8,
                  width=0.5, zorder=3)
    ax.errorbar(['Specialty\nFrame', 'Equity\nFrame'], [m_s, m_e],
                yerr=[1.96*se_s, 1.96*se_e], fmt='none', color=DARK,
                capsize=5, linewidth=1.5, zorder=4)

    ax.set_ylim(1, 5.3)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1\n(Def. eligible)', '2', '3\n(Neutral)', '4', '5\n(Def. NOT eligible)'],
                        fontsize=7.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(label, fontsize=11, fontweight='bold', color=DARK)

    for bar, val in zip(bars, [m_s, m_e]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color=DARK)

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    p_label = f'p = {p_val:.3f} {sig}'
    ax.text(0.5, 0.97, p_label, transform=ax.transAxes,
            ha='center', va='top', fontsize=9,
            color='#15803D' if p_val < 0.05 else GRAY,
            fontweight='bold' if p_val < 0.05 else 'normal')
    ax.text(0.5, 0.89, f't({len(spec_data)+len(eq_data)-2}) = {t_stat:.2f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=8, color=GRAY)

fig4.tight_layout()
fig4.savefig(os.path.join(script_dir, 'fig4_experiment_means.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ─────────────────────────────────────────────
# PRINT SUMMARY STATS TO CONSOLE / TEXT FILE
# ─────────────────────────────────────────────
lines = []
lines.append("=" * 65)
lines.append("SCHOOL VOUCHER SURVEY — SUMMARY STATISTICS")
lines.append("=" * 65)
lines.append(f"Total respondents: {len(df)}\n")

lines.append("─── BLOCK 1: AWARENESS & SUPPORT ───")
for col, order in [('VinState', vin_order), ('VKnow', vknow_order), ('Vsupport', vsupport_order)]:
    lines.append(f"\n{col}  (n={df[col].notna().sum()})")
    pcts = pct_series(df[col].dropna(), order)
    for k, v in pcts.items():
        lines.append(f"  {k:<40} {v:5.1f}%")

lines.append("\n─── BLOCK 2: PERCEIVED EFFECTS ───")
for col, label in effect_vars:
    lines.append(f"\n{col} — {label.strip()}  (n={df[col].notna().sum()})")
    pcts = pct_series(df[col].dropna(), effect_order)
    for k, v in pcts.items():
        lines.append(f"  {k:<40} {v:5.1f}%")

lines.append("\n─── BLOCKS 3 & 4: FRAMING EXPERIMENT (T-TESTS) ───")
lines.append("Scale: 1 = Definitely eligible → 5 = Definitely NOT eligible\n")
for spec_col, eq_col, label in exp_pairs:
    spec_data = df[spec_col + '_num'].dropna()
    eq_data   = df[eq_col + '_num'].dropna()
    t_stat, p_val = stats.ttest_ind(spec_data, eq_data)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    lines.append(f"{label}")
    lines.append(f"  Specialty frame: M = {spec_data.mean():.2f}, SD = {spec_data.std():.2f}, n = {len(spec_data)}")
    lines.append(f"  Equity frame:    M = {eq_data.mean():.2f}, SD = {eq_data.std():.2f}, n = {len(eq_data)}")
    lines.append(f"  t({len(spec_data)+len(eq_data)-2}) = {t_stat:.3f}, p = {p_val:.4f} {sig}")
    lines.append(f"  Direction: {'Equity frame → MORE restriction' if eq_data.mean() > spec_data.mean() else 'Specialty frame → MORE restriction'}\n")

summary_text = "\n".join(lines)
print(summary_text)

with open(os.path.join(script_dir, 'summary_stats.txt'), 'w') as f:
    f.write(summary_text)

print("\nAll outputs saved.")