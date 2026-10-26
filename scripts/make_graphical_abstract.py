"""Create a vector graphical abstract from the frozen numerical results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
R=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':12})
fig=plt.figure(figsize=(13.3,6.7),facecolor='white')
fig.text(.04,.94,'Single-lead AI-ECG across three public cohorts',fontsize=21,weight='bold',color='#162a42')
fig.text(.04,.89,'ECG-derived HYP labels  |  Simulated noise  |  Retrospective evaluation',fontsize=12,color='#46566b')
for x,title,body in [(.04,'DEVELOPMENT','PTB-XL\n21,799 records\nLead I · 100 Hz · 10 s'),(.365,'MODEL','52,225-parameter CNN\nBaseline comparisons\nComponent-level ablation'),(.69,'EXTERNAL VALIDATION','Chapman: n = 2,769\nGeorgia: n = 10,344\nFrozen model weights')]:
 fig.text(x,.80,title,weight='bold',fontsize=12,color='#245f80')
 fig.text(x,.735,body,va='top',linespacing=1.6,fontsize=12)
for x in [.28,.61]: fig.add_artist(FancyArrowPatch((x,.69),(x+.055,.69),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=20,color='#69839a'))
ax=fig.add_axes([.065,.23,.39,.27]);co=['PTB-XL','Chapman','Georgia'];x=range(3)
ax.errorbar([i-.04 for i in x],[.661,.727,.586],yerr=[.0074,.0043,.0030],fmt='o',color='#9b5a45',label='Without augmentation',capsize=3)
ax.errorbar([i+.04 for i in x],[.760,.868,.682],yerr=[.0045,.0019,.0032],fmt='o',color='#1b6a8a',label='Full augmentation',capsize=3)
ax.set_xticks(list(x),co);ax.set_ylim(.5,.95);ax.set_ylabel('AUROC at 4× simulated noise');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.legend(loc='lower left',frameon=False,fontsize=9)
fig.text(.055,.55,'ROBUSTNESS',weight='bold',color='#245f80')
fig.text(.56,.55,'HELD-OUT RECALIBRATION',weight='bold',color='#245f80')
fig.text(.56,.49,'20% calibration → 80% evaluation',fontsize=13)
fig.text(.56,.42,'Mean ECE across 10 stratified splits',fontsize=11,color='#46566b')
fig.text(.56,.35,'Chapman    0.100 → 0.029',fontsize=16,color='#1b6a8a')
fig.text(.56,.28,'Georgia       0.274 → 0.006',fontsize=16,color='#1b6a8a')
fig.text(.04,.115,'Baseline wander carried most of the robustness benefit.',fontsize=13,weight='bold')
fig.text(.04,.06,'No echo-confirmed or real-wearable validation. Chapman calibration reflects case-control prevalence.',fontsize=10,color='#46566b')
for dest in [R/'figures/graphical_abstract',R/'submission_package_JBI/Graphical_Abstract']:
 fig.savefig(dest.with_suffix('.pdf'))
 fig.savefig(dest.with_suffix('.png'),dpi=200)
plt.close(fig)
