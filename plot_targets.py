"""Annual reference, prior and residual distributions for all formal domains."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(ROOT / 'annual_target_statistics.csv')
    assert len(data) == 33
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 6.7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'svg.fonttype': 'none',
        'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'lines.linewidth': 1.2})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.7), sharex=True, sharey='row')
    for col, (domain, title) in enumerate(zip(('GL_OCEAN','GL_LAND','ND'), ('GL ocean','GL land','ND'))):
        rows = data[data.domain == domain].sort_values('year')
        assert rows.year.tolist() == list(range(2015,2026))
        x = rows.year.to_numpy()
        for shift, key, color, label, marker in (
            (-.12,'xco2','#313A43','ACOS reference','o'),
            (.12,'prior','#418C81','Operational prior','s')):
            axes[0,col].errorbar(x+shift,rows[key+'_mean'],yerr=rows[key+'_sd'],
                fmt=marker+'-',color=color,ms=2.6,capsize=1.8,elinewidth=.55,
                label=label,zorder=3)
        axes[1,col].errorbar(x,rows.residual_mean,yerr=rows.residual_sd,
            fmt='o-',color='#80669D',ms=2.8,capsize=1.8,elinewidth=.65,
            label='Reference minus prior',zorder=3)
        axes[1,col].axhline(0,color='#777777',lw=.7,zorder=1)
        axes[0,col].set_title(title,loc='left',pad=10)
        for row in range(2):
            ax=axes[row,col]
            ax.axvspan(2019.62,2020.38,color='#EDF0F2',zorder=0)
            ax.axvline(2020.5,color='#969DA1',lw=.65,ls='--',zorder=1)
            ax.grid(axis='y',color='#E4E6E7',lw=.55,zorder=0)
            ax.set_xlim(2014.5,2025.5)
            ax.set_xticks([2015,2017,2019,2021,2023,2025],
                          ['2015','2017','2019','2021','2023','2025*'],rotation=45)
            ax.text(-.15,1.06,chr(97+row*3+col),transform=ax.transAxes,
                    fontsize=10,fontweight='bold')
        axes[1,col].set_xlabel('Year')
    axes[0,0].set_ylabel('XCO₂ (ppm)')
    axes[1,0].set_ylabel('XCO₂ − prior (ppm)')
    axes[0,0].set_ylim(390,434)
    axes[0,0].set_yticks([390,400,410,420,430])
    axes[1,0].set_ylim(-6,4)
    axes[1,0].set_yticks([-6,-4,-2,0,2,4])
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',
               bbox_to_anchor=(.54,1.01),ncol=2,frameon=False)
    fig.subplots_adjust(left=.09,right=.99,top=.88,bottom=.14,hspace=.32,wspace=.18)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for text in fig.findobj(match=matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            assert box.x0 >= -5 and box.x1 <= fig.bbox.width+5, text.get_text()
    folder=ROOT/'figures'
    folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg','tiff'):
        fig.savefig(folder/f'Fig5_three_domain_targets.{ext}',dpi=450,
                    bbox_inches='tight',pad_inches=.07)
    fig.savefig(folder/'Fig5_three_domain_targets_preview.png',dpi=170,
                bbox_inches='tight',pad_inches=.07)
    plt.close(fig)


if __name__ == '__main__':
    main()
