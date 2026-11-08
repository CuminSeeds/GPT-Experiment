"""用 Matplotlib 默认风格绘制位置编码与头数对比。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_font import apply_report_font

apply_report_font()

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'


def read_history(name, stage):
    folder = ROOT / 'results' if name == 'sinusoidal_h4' else ROOT / 'improvements' / name / 'results'
    h = json.loads((folder / f'{stage}_history.json').read_text(encoding='utf-8'))
    return h if stage == 'pretrain' else h[1:]


def draw(group, stage, entries):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    key = 'step' if stage == 'pretrain' else 'epoch'
    for i, (name, label) in enumerate(entries):
        h = read_history(name, stage)
        x = [r[key] for r in h]
        for split, suffix, style in [('train', 'train', '--'), ('val', 'val', '-')]:
            axes[0].plot(x, [r[split]['loss'] for r in h], color=f'C{i}',
                         linestyle=style, label=f'{label} {suffix}')
            axes[1].plot(x, [100*r[split]['accuracy'] for r in h], color=f'C{i}',
                         linestyle=style)
    axes[0].set_ylabel('Cross-entropy loss')
    axes[1].set_ylabel('Accuracy (%)')
    axes[0].legend(ncol=2)
    for ax in axes:
        ax.set_xlabel('Training step' if stage == 'pretrain' else 'Epoch')
    fig.tight_layout()
    path = OUT / f'{group}_{stage}_comparison.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(path)


def main():
    OUT.mkdir(exist_ok=True)
    groups = {
        'position': [('sinusoidal_h4', 'Sin'), ('rope_h4', 'RoPE')],
        'heads': [('rope_h2', 'H2'), ('rope_h4', 'H4'), ('rope_h8', 'H8')],
    }
    for group, entries in groups.items():
        for stage in ('pretrain', 'finetune'):
            draw(group, stage, entries)


if __name__ == '__main__':
    main()
