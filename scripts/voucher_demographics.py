import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── PATHS ──────────────────────────────────────────────────────────────────
DATA_PATH  = '/Users/lena/polsreserachdepartment/POLSLab_Fall25_National_Labels_with_Background.xlsx'
OUTPUT_DIR = '/Users/lena/polsreserachdepartment/'
# ────────────────────────────────────────────────────────────────────────────

df = pd.read_excel(DATA_PATH)

# ─── Encode numeric outcomes ─────────────────────────────────────────────────
support_map = {'Strongly oppose':1,'Somewhat oppose':2,'Not sure or neutral':3,
               'Somewhat support':4,'Strongly support':5}
df['Vsupport_n'] = df['Vsupport'].map(support_map)

effect_map = {'They are greatly harmed':1,'They are slightly harmed':2,
              'No effect or not sure':3,'They somewhat benefit':4,'They greatly benefit':5}
for col in ['VPoor','VWealthy','VReligious','VUrban','VRural','Vswd']:
    df[col+'_n'] = df[col].map(effect_map)

rule_map = {'Yes, the school should definitely be eligible for vouchers':1,
            'It probably should be eligible':2,'Neutral or not sure':3,
            'It probably should not be eligible':4,
            'No, it definitely should not be eligible for vouchers':5}
for col in ['TuitionRule','ReligionRule','SWDrule','TuitionRuleE','ReligionRuleE','SWDruleE']:
    df[col+'_n'] = df[col].map(rule_map)

outcomes_pop = ['Vsupport_n','VPoor_n','VWealthy_n','VReligious_n','VUrban_n','VRural_n','Vswd_n']
labels_pop   = ['General\nSupport','Poor\nFamilies','Wealthy\nFamilies','Religious\nFamilies',
                'Urban\nFamilies','Rural\nFamilies','Students\nw/ Disabilities']
school_pairs = [
    ('TuitionRule_n','TuitionRuleE_n','High-Tuition School'),
    ('ReligionRule_n','ReligionRuleE_n','Religious School'),
    ('SWDrule_n','SWDruleE_n','Selective Cognitive School'),
]

# ─── Demographic groupings ────────────────────────────────────────────────────

# Race (simplified)
race_map = {
    'White': 'White',
    'Black or African American': 'Black/African Am.',
    'Chinese': 'Asian',
    'Vietnamese': 'Asian',
    'Asian Indian': 'Asian',
    'Filipino': 'Asian',
    'Korean': 'Asian',
    'Japanese': 'Asian',
    'American Indian or Alaska Native': 'Indigenous',
    'An ethnicity not listed here': 'Other',
}
df['race_grp'] = df['Race'].map(race_map).fillna('Other')
# Also flag Hispanic
df['hispanic'] = df['Ethnicity'].str.startswith('Yes', na=False)
df.loc[df['hispanic'], 'race_grp'] = 'Hispanic/Latino'

# Gender
df['gender_grp'] = df['Gender'].map({'Man':'Man','Woman':'Woman'})

# Age groups
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
bins  = [17, 29, 44, 59, 120]
labels_age = ['18-29', '30-44', '45-59', '60+']
df['age_grp'] = pd.cut(df['Age'], bins=bins, labels=labels_age)

# Income groups (collapse into 5 buckets)
def income_bucket(x):
    if pd.isna(x) or x == 'Prefer not to say': return np.nan
    low = ['Less than $10,000','$10,000-$19,999','$20,000-$29,999']
    lowmid = ['$30,000-$39,999','$40,000-$49,999','$50,000-$59,999']
    mid = ['$60,000-$69,999','$70,000-$79,999','$80,000-$89,999','$90,000-$99,999']
    highmid = ['$100,000-$124,999','$125,000-$149,999']
    high = ['$150,000-$174,999','$175,000-$199,999','$200,000-$224,999','$225,000-$249,999','$250,000 or more']
    if x in low: return 'Low\n(<$30k)'
    if x in lowmid: return 'Low-Mid\n($30-59k)'
    if x in mid: return 'Middle\n($60-99k)'
    if x in highmid: return 'High-Mid\n($100-149k)'
    if x in high: return 'High\n($150k+)'
    return np.nan
df['income_grp'] = df['Household Income'].apply(income_bucket)

# Education (collapsed)
def edu_bucket(x):
    if pd.isna(x) or x == 'Prefer not to say': return np.nan
    if x in ['No formal education','Less than a high school diploma']: return 'No HS Diploma'
    if 'High school' in str(x): return 'HS Diploma/GED'
    if 'Some college' in str(x) or 'Associate' in str(x): return 'Some College\n/Associate'
    if "Bachelor" in str(x): return "Bachelor's"
    if "Master" in str(x): return "Master's"
    if "Professional" in str(x) or "Doctorate" in str(x): return 'Graduate\nDegree'
    return np.nan
df['edu_grp'] = df['Education'].apply(edu_bucket)

print("Demographic groups ready.")
for col in ['race_grp','gender_grp','age_grp','income_grp','edu_grp']:
    print(f"\n{col}:\n{df[col].value_counts().to_string()}")

# ─── STYLE ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#F9F7F4',
    'axes.facecolor': '#F9F7F4',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

def make_subgroup_fig(df, group_col, group_order, group_colors, title, xlabel_rotation=0):
    """Bar chart grid: each outcome panel, bars = subgroups, with ANOVA stats."""
    data = df[df[group_col].isin(group_order)].copy()
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor('#F9F7F4')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.01)
    n = len(group_order)
    for i, (outcome, label) in enumerate(zip(outcomes_pop, labels_pop)):
        ax = axes[i // 4][i % 4]
        means, sems = [], []
        for g in group_order:
            grp = data[data[group_col] == g][outcome].dropna()
            means.append(grp.mean()); sems.append(grp.sem())
        ax.bar(range(n), means, color=group_colors[:n], edgecolor='white', linewidth=1.2, width=0.65)
        ax.errorbar(range(n), means, yerr=[1.96*s for s in sems],
                    fmt='none', color='#333', capsize=3, linewidth=1.2)
        ax.set_xticks(range(n))
        short_labels = [g.replace('\n',' ') if len(g) < 14 else g for g in group_order]
        ax.set_xticklabels(short_labels, fontsize=7.5, rotation=xlabel_rotation, ha='right' if xlabel_rotation else 'center')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axhline(3, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        ax.set_ylim(1, 5)
        ax.set_ylabel('Score (1-5)', fontsize=8)
        vals = [data[data[group_col] == g][outcome].dropna() for g in group_order]
        f, p = stats.f_oneway(*vals)
        sig = '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'n.s.'))
        ax.set_xlabel(f'F={f:.1f}, {sig}', fontsize=7.5, color='#555')
        # add mean values on bars
        for xi, m in enumerate(means):
            ax.text(xi, 0.05 + 1, f'{m:.2f}', ha='center', va='bottom', fontsize=7, color='white', fontweight='bold')
    axes[1][3].set_visible(False)
    plt.tight_layout()
    return fig

def make_framing_fig(df, group_col, group_order, group_colors, title, xlabel_rotation=0):
    """Grouped bar (spec vs equity) × subgroups, for each school type."""
    data = df[df[group_col].isin(group_order)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor('#F9F7F4')
    fig.suptitle(title + '\n(Higher = more restrictive)', fontsize=14, fontweight='bold')
    n = len(group_order)
    x = np.arange(n)
    width = 0.35
    for ax, (spec_col, eq_col, stitle) in zip(axes, school_pairs):
        spec_means, eq_means, spec_sems, eq_sems = [], [], [], []
        for g in group_order:
            grp = data[data[group_col] == g]
            s = grp[spec_col].dropna(); e = grp[eq_col].dropna()
            spec_means.append(s.mean()); spec_sems.append(s.sem())
            eq_means.append(e.mean()); eq_sems.append(e.sem())
        ax.bar(x - width/2, spec_means, width, label='Specialty Frame', color='#74ADD1', edgecolor='white')
        ax.bar(x + width/2, eq_means, width, label='Equity Frame', color='#D6604D', edgecolor='white')
        ax.errorbar(x - width/2, spec_means, yerr=[1.96*s for s in spec_sems], fmt='none', color='#333', capsize=3)
        ax.errorbar(x + width/2, eq_means, yerr=[1.96*s for s in eq_sems], fmt='none', color='#333', capsize=3)
        ax.set_xticks(x)
        short_labels = [g.replace('\n',' ') for g in group_order]
        ax.set_xticklabels(short_labels, fontsize=8, rotation=xlabel_rotation, ha='right' if xlabel_rotation else 'center')
        ax.set_title(stitle, fontsize=11, fontweight='bold')
        ax.set_ylim(1, 5); ax.set_ylabel('Restriction (1-5)', fontsize=9)
        ax.axhline(3, color='gray', linestyle=':', alpha=0.5)
        if ax == axes[0]: ax.legend(fontsize=8)
    plt.tight_layout()
    return fig

# ─── RACE ────────────────────────────────────────────────────────────────────
race_order  = ['White','Black/African Am.','Hispanic/Latino','Asian','Other']
race_colors = ['#D9D9D9','#4393C3','#74C476','#FDB863','#BCBDDC']

fig = make_subgroup_fig(df, 'race_grp', race_order, race_colors, 'Voucher Attitudes by Race/Ethnicity')
plt.savefig(OUTPUT_DIR + 'fig_race_outcomes.png', dpi=150, bbox_inches='tight'); plt.close()
fig = make_framing_fig(df, 'race_grp', race_order, race_colors, 'Framing Experiment by Race/Ethnicity')
plt.savefig(OUTPUT_DIR + 'fig_race_framing.png', dpi=150, bbox_inches='tight'); plt.close()
print("Race figs done.")

# ─── GENDER ──────────────────────────────────────────────────────────────────
gender_order  = ['Man','Woman']
gender_colors = ['#4393C3','#F4A582']

fig = make_subgroup_fig(df, 'gender_grp', gender_order, gender_colors, 'Voucher Attitudes by Gender')
plt.savefig(OUTPUT_DIR + 'fig_gender_outcomes.png', dpi=150, bbox_inches='tight'); plt.close()
fig = make_framing_fig(df, 'gender_grp', gender_order, gender_colors, 'Framing Experiment by Gender')
plt.savefig(OUTPUT_DIR + 'fig_gender_framing.png', dpi=150, bbox_inches='tight'); plt.close()
print("Gender figs done.")

# ─── AGE GROUP ───────────────────────────────────────────────────────────────
age_order  = ['18-29','30-44','45-59','60+']
age_colors = ['#FDCC8A','#FC8D59','#E34A33','#B30000']

# Convert age_grp from Categorical to string for filtering
df['age_grp'] = df['age_grp'].astype(str)
df.loc[df['age_grp'] == 'nan', 'age_grp'] = np.nan

fig = make_subgroup_fig(df, 'age_grp', age_order, age_colors, 'Voucher Attitudes by Age Group')
plt.savefig(OUTPUT_DIR + 'fig_age_outcomes.png', dpi=150, bbox_inches='tight'); plt.close()
fig = make_framing_fig(df, 'age_grp', age_order, age_colors, 'Framing Experiment by Age Group')
plt.savefig(OUTPUT_DIR + 'fig_age_framing.png', dpi=150, bbox_inches='tight'); plt.close()
print("Age figs done.")

# ─── INCOME ──────────────────────────────────────────────────────────────────
income_order  = ['Low\n(<$30k)','Low-Mid\n($30-59k)','Middle\n($60-99k)','High-Mid\n($100-149k)','High\n($150k+)']
income_colors = ['#EFF3FF','#9ECAE1','#3182BD','#08519C','#08306B']

fig = make_subgroup_fig(df, 'income_grp', income_order, income_colors,
                        'Voucher Attitudes by Household Income', xlabel_rotation=20)
plt.savefig(OUTPUT_DIR + 'fig_income_outcomes.png', dpi=150, bbox_inches='tight'); plt.close()
fig = make_framing_fig(df, 'income_grp', income_order, income_colors,
                       'Framing Experiment by Household Income', xlabel_rotation=20)
plt.savefig(OUTPUT_DIR + 'fig_income_framing.png', dpi=150, bbox_inches='tight'); plt.close()
print("Income figs done.")

# ─── EDUCATION ───────────────────────────────────────────────────────────────
edu_order  = ['No HS Diploma','HS Diploma/GED','Some College\n/Associate',"Bachelor's","Master's",'Graduate\nDegree']
edu_colors = ['#FEEDDE','#FDBE85','#FD8D3C','#E6550D','#A63603','#7F2704']

fig = make_subgroup_fig(df, 'edu_grp', edu_order, edu_colors,
                        'Voucher Attitudes by Education Level', xlabel_rotation=20)
plt.savefig(OUTPUT_DIR + 'fig_edu_outcomes.png', dpi=150, bbox_inches='tight'); plt.close()
fig = make_framing_fig(df, 'edu_grp', edu_order, edu_colors,
                       'Framing Experiment by Education Level', xlabel_rotation=20)
plt.savefig(OUTPUT_DIR + 'fig_edu_framing.png', dpi=150, bbox_inches='tight'); plt.close()
print("Education figs done.")

# ─── DEMOGRAPHIC SUMMARY HEATMAP ─────────────────────────────────────────────
all_outcomes = outcomes_pop + ['TuitionRule_n','ReligionRule_n','SWDrule_n',
                                'TuitionRuleE_n','ReligionRuleE_n','SWDruleE_n']
all_labels_h = labels_pop + ['Tuition\n(Spec)','Religion\n(Spec)','SWD\n(Spec)',
                              'Tuition\n(Eq)','Religion\n(Eq)','SWD\n(Eq)']

rows, row_labels, row_sections = [], [], []

groups = [
    ('race_grp', race_order, 'RACE'),
    ('gender_grp', gender_order, 'GENDER'),
    ('age_grp', age_order, 'AGE'),
    ('income_grp', income_order, 'INCOME'),
    ('edu_grp', edu_order, 'EDUCATION'),
]

section_starts = {}
idx = 0
for group_col, group_order, section_name in groups:
    section_starts[section_name] = idx
    data = df[df[group_col].isin(group_order)]
    for g in group_order:
        grp = data[data[group_col] == g]
        rows.append([grp[o].mean() for o in all_outcomes])
        row_labels.append(g.replace('\n',' '))
        idx += 1

matrix = np.array(rows)

fig, ax = plt.subplots(figsize=(20, 14))
fig.patch.set_facecolor('#F9F7F4')
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=1.8, vmax=4.2)
ax.set_xticks(range(len(all_labels_h)))
ax.set_xticklabels(all_labels_h, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=9)

for i in range(len(row_labels)):
    for j in range(len(all_labels_h)):
        val = matrix[i, j]
        color = 'white' if val < 2.2 or val > 3.8 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7.5, color=color, fontweight='bold')

# Section dividers and labels
boundaries = {'RACE': 0, 'GENDER': 5, 'AGE': 7, 'INCOME': 11, 'EDUCATION': 16}
section_ends = {'RACE': 4.5, 'GENDER': 6.5, 'AGE': 10.5, 'INCOME': 15.5}
for name, end in section_ends.items():
    ax.axhline(end, color='black', linewidth=1.5)

# Section labels on right margin
section_midpoints = {'RACE': 2, 'GENDER': 6, 'AGE': 9, 'INCOME': 13, 'EDUCATION': 18.5}
for name, mid in section_midpoints.items():
    ax.text(len(all_labels_h) - 0.3, mid, name, ha='left', va='center',
            fontsize=9, fontweight='bold', color='#333')

ax.axvline(6.5, color='black', linewidth=1.5, alpha=0.4)

cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.08)
cbar.set_label('Low ← 1─────────────────5 → High', fontsize=9)
ax.set_title('Voucher Survey: Mean Scores by Demographic Group\n(Green = Higher support/perceived benefit  |  Red = Lower support/more restrictive)',
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'fig_demo_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Demographic heatmap done.")

# ─── STATS SUMMARY ───────────────────────────────────────────────────────────
lines = []
lines.append("=" * 70)
lines.append("DEMOGRAPHIC SUBGROUP ANALYSIS — VOUCHER SURVEY (Fall 2025, N=1,005)")
lines.append("=" * 70)
lines.append("Income classification: Low=<$30k | Low-Mid=$30-59k | Middle=$60-99k")
lines.append("                       High-Mid=$100-149k | High=$150k+")
lines.append("Age groups: 18-29 | 30-44 | 45-59 | 60+")

demo_groups = [
    ('race_grp', race_order, 'RACE'),
    ('gender_grp', gender_order, 'GENDER'),
    ('age_grp', age_order, 'AGE GROUP'),
    ('income_grp', income_order, 'HOUSEHOLD INCOME'),
    ('edu_grp', edu_order, 'EDUCATION'),
]

for group_col, group_order, section_title in demo_groups:
    lines.append(f"\n{'─'*66}")
    lines.append(f"SECTION: {section_title}  (ANOVA on each outcome)")
    lines.append(f"{'─'*66}")
    data = df[df[group_col].isin(group_order)]
    for outcome, label in zip(outcomes_pop, labels_pop):
        vals = [data[data[group_col] == g][outcome].dropna() for g in group_order]
        f, p = stats.f_oneway(*vals)
        ms = [v.mean() for v in vals]
        sig = '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'n.s.'))
        lines.append(f"\n{label.replace(chr(10),' ')}: F={f:.2f}, p={p:.4f} {sig}")
        for gname, m in zip(group_order, ms):
            gname_clean = gname.replace('\n', ' ')
            lines.append(f"   {gname_clean:25s}: M = {m:.3f}")
        lines.append(f"   >> Range: {min(ms):.3f} to {max(ms):.3f}, Δ = {max(ms)-min(ms):.3f}")

    lines.append(f"\n  Framing Experiment Δ (Equity - Specialty):")
    for spec_col, eq_col, stitle in school_pairs:
        lines.append(f"  {stitle}:")
        for gname in group_order:
            g = data[data[group_col] == gname]
            s = g[spec_col].dropna().mean(); e = g[eq_col].dropna().mean()
            lines.append(f"    {gname.replace(chr(10),' '):25s}: Spec={s:.2f}, Equity={e:.2f}, Δ={e-s:+.2f}")

lines.append("\n*** p<.001  ** p<.01  * p<.05  n.s. = not significant")
lines.append("=" * 70)

with open(OUTPUT_DIR + 'demographic_stats.txt', 'w') as f:
    f.write('\n'.join(lines))
print("Stats file done.")