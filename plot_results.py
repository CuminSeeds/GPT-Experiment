"""全部图形采用 Matplotlib 默认风格。"""
from pathlib import Path
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_font import apply_report_font

apply_report_font()

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'


def draw(stage):
    OUT.mkdir(exist_ok=True)
    h=json.loads((ROOT/'results'/f'{stage}_history.json').read_text(encoding='utf-8'))
    # 第0轮的微调损失很大，后续曲线从第1轮开始，便于观察过拟合。
    # 第0轮数值仍完整保存在 JSON 记录与报告中。
    if stage=='finetune': h=h[1:]
    key='step' if stage=='pretrain' else 'epoch'
    x=[r[key] for r in h]
    fig,ax=plt.subplots(1,2,figsize=(10,3.6))
    for split,label in [('train','Training'),('val','Validation')]:
        ax[0].plot(x,[r[split]['loss'] for r in h],label=label)
        ax[1].plot(x,[r[split]['accuracy']*100 for r in h],label=label)
    for a in ax:
        a.set_xlabel('Training step' if key=='step' else 'Epoch')
        a.legend()
    ax[0].set_ylabel('Cross-entropy loss')
    ax[1].set_ylabel('Accuracy (%)')
    fig.tight_layout()
    fig.savefig(OUT/f'{stage}_curves.png',dpi=200)
    plt.close(fig)
    print(OUT/f'{stage}_curves.png')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pretrain','finetune','all'],default='all')
    a=p.parse_args()
    for stage in ['pretrain','finetune']:
        if a.stage in (stage,'all'): draw(stage)
