import os
import shutil

# ─── SET THIS TO YOUR FOLDER ─────────────────────────────────────────────────
BASE = '/Users/lena/polsreserachdepartment'
# ─────────────────────────────────────────────────────────────────────────────

# Folder structure to create
folders = [
    'figures/original',
    'figures/subgroups',
    'figures/demographics',
    'scripts',
    'data',
    'stats',
]

for folder in folders:
    os.makedirs(os.path.join(BASE, folder), exist_ok=True)

# ─── FILE MAPPING ─────────────────────────────────────────────────────────────
moves = {
    # Original analysis figures (Figure 1-4)
    'Figure 1 - Awareness & Support.png':       'figures/original',
    'Figure 2 - Percieved Effects.png':          'figures/original',
    'Figure 3 - Speciality vs. Equity.png':      'figures/original',
    'Figure 4- Mean Eligibility Scores.png':     'figures/original',

    # Subgroup figures
    'fig_party_outcomes.png':                    'figures/subgroups',
    'fig_party_framing.png':                     'figures/subgroups',
    'fig_relig_outcomes.png':                    'figures/subgroups',
    'fig_relig_framing.png':                     'figures/subgroups',
    'fig_sped_outcomes.png':                     'figures/subgroups',
    'fig_sped_framing.png':                      'figures/subgroups',
    'fig_heatmap.png':                           'figures/subgroups',

    # Demographic figures
    'fig_race_outcomes.png':                     'figures/demographics',
    'fig_race_framing.png':                      'figures/demographics',
    'fig_gender_outcomes.png':                   'figures/demographics',
    'fig_gender_framing.png':                    'figures/demographics',
    'fig_age_outcomes.png':                      'figures/demographics',
    'fig_age_framing.png':                       'figures/demographics',
    'fig_income_outcomes.png':                   'figures/demographics',
    'fig_income_framing.png':                    'figures/demographics',
    'fig_edu_outcomes.png':                      'figures/demographics',
    'fig_edu_framing.png':                       'figures/demographics',
    'fig_demo_heatmap.png':                      'figures/demographics',

    # Scripts
    'analysis.py':                               'scripts',
    'voucher_subgroup.py':                       'scripts',
    'voucher_demographics.py':                   'scripts',
    'organize_folder.py':                        'scripts',

    # Data
    'POLSLab_Fall25_National_Labels_with_Background.xlsx': 'data',

    # Stats / text files
    'Summary_Statistics.txt':                    'stats',
    'Subgroups_Statistics.txt':                  'stats',
    'subgroup_stats.txt':                        'stats',
    'demographic_stats.txt':                     'stats',
}

moved, skipped = [], []

for filename, dest_folder in moves.items():
    src = os.path.join(BASE, filename)
    dst = os.path.join(BASE, dest_folder, filename)
    if os.path.exists(src):
        shutil.move(src, dst)
        moved.append(f'  ✓  {filename}  →  {dest_folder}/')
    else:
        skipped.append(f'  –  {filename} (not found, skipped)')

print("=" * 60)
print("MOVED:")
for m in moved: print(m)
if skipped:
    print("\nSKIPPED (file not found):")
    for s in skipped: print(s)
print("=" * 60)
print("\nDone! New structure:")
for root, dirs, files in os.walk(BASE):
    dirs[:] = sorted([d for d in dirs if not d.startswith('.')])
    level = root.replace(BASE, '').count(os.sep)
    indent = '  ' * level
    folder_name = os.path.basename(root) or 'polsreserachdepartment'
    print(f'{indent}{folder_name}/')
    for f in sorted(files):
        if not f.startswith('.'):
            print(f'{indent}  {f}')