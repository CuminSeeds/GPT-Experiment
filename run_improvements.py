"""仅比较位置编码和注意力头数；原始实验文件不覆盖。"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import json
import torch
import train
from model import Config, TinyGPT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'improvements'
VARIANTS = {
    'rope_h4': Config(position_encoding='rope', heads=4),
    'rope_h2': Config(position_encoding='rope', heads=2),
    'rope_h8': Config(position_encoding='rope', heads=8),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize():
    rows = []
    sources = [('sinusoidal_h4', ROOT / 'results', Config())]
    sources += [(name, OUT / name / 'results', cfg) for name, cfg in VARIANTS.items()]
    for name, folder, cfg in sources:
        if not (folder / 'finetune_metrics.json').exists():
            continue
        lm = read(folder / 'pretrain_metrics.json')
        ft = read(folder / 'finetune_metrics.json')
        rows.append(dict(name=name, config=asdict(cfg), parameters=628226,
                         pretrain=lm, finetune=ft))
    result = dict(
        design='正弦余弦4头与RoPE4头比较位置编码；RoPE固定时比较2、4、8头。',
        baseline='复用改进前已经完成的正弦余弦4头实验；相同数据、种子和训练设置。',
        selection='各模型按验证损失选择权重；最终配置按短信验证损失选择，不按测试指标选择。',
        repetitions=1, models=rows)
    if len(rows) == 4:
        selected = min(rows, key=lambda r: r['finetune']['val']['loss'])
        result['selected_by_validation'] = selected['name']
        result['selected_validation_loss'] = selected['finetune']['val']['loss']
    (OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPARISON', json.dumps(result, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['all', *VARIANTS], default='all')
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.summarize:
        summarize()
        return
    if train.DEVICE != 'cuda':
        print('CUDA 不可用，将使用 CPU 完整训练。', flush=True)
    files = [ROOT / 'data' / name for name in (
        'lm_train.txt', 'lm_val.txt', 'lm_test.txt', 'sms_train.json', 'sms_val.json', 'sms_test.json')]
    manifest = dict(settings=train.SETTINGS,
                    data_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                    variants={name: asdict(cfg) for name, cfg in VARIANTS.items()})
    (OUT / 'settings.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    targets = VARIANTS if args.variant == 'all' else [args.variant]
    for name in targets:
        folder = OUT / name
        train.RESULTS, train.WEIGHTS = folder / 'results', folder / 'weights'
        train.RESULTS.mkdir(parents=True, exist_ok=True)
        train.WEIGHTS.mkdir(parents=True, exist_ok=True)
        cfg = VARIANTS[name]
        train.seed_all(42)
        count = sum(p.numel() for p in TinyGPT(cfg).parameters())
        assert count == 628226
        print('VARIANT_START', name, json.dumps(asdict(cfg)), flush=True)
        if not (train.RESULTS / 'pretrain_metrics.json').exists():
            train.pretrain(cfg)
        if not (train.RESULTS / 'finetune_metrics.json').exists():
            train.finetune()
        print('VARIANT_DONE', name, flush=True)
    summarize()


if __name__ == '__main__':
    main()
