#!/usr/bin/env python3
"""Regenerate Figure 4: Devign placebo on commit-composition strata.
Run anywhere with matplotlib; drop the PDF/PNG into the manuscript."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

regimes = ['Shipped-split\ntraining', 'Commit-clustered\ntraining']
acc  = {'multi-commit': [0.695, 0.626], 'singleton': [0.547, 0.569]}
sd   = {'multi-commit': [0.006, 0.061], 'singleton': [0.009, 0.020]}
base = {'multi-commit': [0.585, 0.697], 'singleton': [0.553, 0.566]}

x = np.arange(2); w = 0.34
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for k, (name, color) in enumerate([('multi-commit', '#4472a8'), ('singleton', '#b0b0b0')]):
    xs = x + (k - 0.5) * w
    ax.bar(xs, acc[name], w * 0.92, yerr=sd[name], capsize=3, color=color,
           label=f'{name} stratum', edgecolor='black', linewidth=0.6)
    for xi, b in zip(xs, base[name]):
        ax.hlines(b, xi - w * 0.46, xi + w * 0.46, colors='black',
                  linestyles=(0, (3, 2)), linewidth=1.4)
ax.hlines([], [], [], colors='black', linestyles=(0, (3, 2)), label='stratum majority rate')
ax.set_xticks(x); ax.set_xticklabels(regimes)
ax.set_ylabel('Accuracy'); ax.set_ylim(0.45, 0.78)
ax.legend(frameon=False, fontsize=8, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('figure4_placebo.pdf'); fig.savefig('figure4_placebo.png', dpi=300)
print('wrote figure4_placebo.pdf/.png')
print('reading: skill above dashed line only in shipped-split multi-commit bar')