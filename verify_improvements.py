"""核对验证集选择规则，并在 CPU 上重新加载三组改进模型。"""
from pathlib import Path
import hashlib
import json
import torch
import train
from model import load_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'improvements'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(4)
    train.DEVICE = 'cpu'
    manifest = read(OUT / 'settings.json')
    for name, expected in manifest['data_sha256'].items():
        assert hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() == expected
    comparison = read(OUT / 'comparison.json')
    selected = min(comparison['models'], key=lambda r: r['finetune']['val']['loss'])
    assert selected['name'] == comparison['selected_by_validation']
    tensors = train.sms_tensors(read(ROOT / 'data' / 'sms_test.json'))
    output = []
    for row in comparison['models']:
        name = row['name']
        folder = ROOT if name == 'sinusoidal_h4' else OUT / name
        ph = read(folder / 'results' / 'pretrain_history.json')
        fh = read(folder / 'results' / 'finetune_history.json')
        assert len(ph) == 19 and len(fh) == 21
        assert min(ph[1:], key=lambda r: r['val']['loss'])['step'] == row['pretrain']['best_step']
        assert min(fh[1:], key=lambda r: r['val']['loss'])['epoch'] == row['finetune']['best_epoch']
        model = load_model(folder / 'weights' / 'finetuned.pt', 'cpu')
        assert sum(p.numel() for p in model.parameters()) == 628226
        metrics, predictions, raw = train.sms_eval(model, tensors, return_predictions=True)
        saved = read(folder / 'results' / 'test_predictions.json')
        matches = sum(p == r['prediction'] for p,r in zip(predictions,saved))
        assert matches == len(saved) == 517
        for key in ('accuracy','spam_precision','spam_recall','spam_f1'):
            assert metrics[key] == row['finetune']['test'][key]
        assert abs(metrics['loss'] - row['finetune']['test']['loss']) < 1e-4
        output.append(dict(name=name, cpu_prediction_matches=matches,
                           selected_pretrain_step=row['pretrain']['best_step'],
                           selected_finetune_epoch=row['finetune']['best_epoch'],
                           cpu_loss=metrics['loss'], gpu_loss=row['finetune']['test']['loss']))
    result = dict(data_hashes_match=True, validation_selection_verified=True,
                  selected_by_validation=selected['name'], reload_checks=output)
    (OUT / 'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
