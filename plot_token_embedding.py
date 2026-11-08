"""分词与词嵌入对比曲线。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_font import apply_report_font

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def draw(stage):
    apply_report_font()
    entries=[('mixed_random','Mixed'),('mixed_cbow','Mixed+CBOW')]
    if stage=='finetune':entries.insert(0,('character_random','Char'))
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    for i,(name,label) in enumerate(entries):
        folder=ROOT/'improvements/rope_h4' if name=='character_random' else ROOT/'token_embedding'/name
        h=read(folder/'results'/f'{stage}_history.json')
        if stage=='finetune':h=h[1:]
        x=[r['step' if stage=='pretrain' else 'epoch'] for r in h]
        for split,suffix,style in [('train','train','--'),('val','val','-')]:
            axes[0].plot(x,[r[split]['loss'] for r in h],color=f'C{i}',linestyle=style,label=f'{label} {suffix}')
            axes[1].plot(x,[r[split]['accuracy']*100 for r in h],color=f'C{i}',linestyle=style)
    axes[0].set_ylabel('Token cross-entropy loss' if stage=='pretrain' else 'Cross-entropy loss')
    axes[1].set_ylabel('Token accuracy (%)' if stage=='pretrain' else 'Accuracy (%)')
    axes[0].legend(ncol=2)
    for ax in axes:ax.set_xlabel('Training step' if stage=='pretrain' else 'Epoch')
    fig.tight_layout();path=ROOT/'figures'/f'token_embedding_{stage}.png'
    fig.savefig(path,dpi=220);plt.close(fig);print(path)

if __name__=='__main__':
    for stage in ['pretrain','finetune']:draw(stage)
