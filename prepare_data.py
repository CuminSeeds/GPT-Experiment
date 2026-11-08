"""下载公开数据，固定划分。去重和划分不使用模型或测试结果。"""
from pathlib import Path
import hashlib, json, random, subprocess, unicodedata, urllib.request, zipfile
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def clean_sms(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', errors='ignore').decode('ascii')
    return ' '.join(text.split())


def prompt_for(text, context=192):
    prefix, suffix = 'Message: ', '\nLabel:'
    return prefix + text[:context - len(prefix) - len(suffix)] + suffix


def prepare():
    DATA.mkdir(exist_ok=True)
    sources = {
        'tinyshakespeare.txt': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'sms.zip': 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',
    }
    for name, url in sources.items():
        if not (DATA/name).exists():
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            temporary = DATA/(name+'.download')
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    temporary.write_bytes(r.read())
            except (OSError, TimeoutError):
                subprocess.run(['curl.exe' if __import__('os').name=='nt' else 'curl',
                                '-fL','--retry','2','--connect-timeout','20','--max-time','180',url,
                                '-o',str(temporary)], check=True)
            temporary.replace(DATA/name)
    text = (DATA/'tinyshakespeare.txt').read_text(encoding='utf-8')
    n = len(text)
    lm = {'train': text[:int(.9*n)], 'val': text[int(.9*n):int(.95*n)], 'test': text[int(.95*n):]}
    for split, part in lm.items():
        (DATA/f'lm_{split}.txt').write_text(part, encoding='utf-8')
    with zipfile.ZipFile(DATA/'sms.zip') as z:
        raw = z.read('SMSSpamCollection').decode('utf-8')
        (DATA/'sms_readme.txt').write_bytes(z.read('readme'))
    # 按模型实际可见的前176个字符去重，避免不同长短信截断后跨集合重复。
    # 保留组内第一条完整短信以便查看原文；矛盾标注整组删除。
    groups = {}
    raw_counts = Counter()
    for line in raw.splitlines():
        label, message = line.split('\t', 1)
        raw_counts[label] += 1
        s = clean_sms(message)
        group=groups.setdefault(s[:176], {'text':s, 'labels':set()})
        group['labels'].add(int(label == 'spam'))
    records = [{'text': g['text'], 'label': next(iter(g['labels']))} for g in groups.values() if len(g['labels']) == 1]
    rng = random.Random(2026)
    splits = {k: [] for k in ('train', 'val', 'test')}
    for label in (0, 1):
        rows = [r for r in records if r['label'] == label]
        rng.shuffle(rows)
        a, b = int(.8*len(rows)), int(.9*len(rows))
        for k, vals in zip(splits, (rows[:a], rows[a:b], rows[b:])):
            splits[k].extend(vals)
    for k, rows in splits.items():
        rng.shuffle(rows)
        (DATA/f'sms_{k}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    sets = [set(prompt_for(r['text']) for r in splits[k]) for k in splits]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
    summary = {
        'source_urls': sources,
        'sha256': {n: hashlib.sha256((DATA/n).read_bytes()).hexdigest() for n in sources},
        'lm_characters': {k: len(v) for k,v in lm.items()},
        'lm_non_ascii': sum(ord(c) >= 128 for c in text),
        'sms_raw_rows': sum(raw_counts.values()), 'sms_raw_class_counts': dict(raw_counts),
        'sms_unique_rows': len(records), 'sms_conflicting_groups_removed': sum(len(g['labels'])>1 for g in groups.values()),
        'sms_deduplication_key': '清洗后模型可见的短信前176字符', 'cross_split_identical_prompts': 0,
        'sms_splits': {k: {'total': len(v), 'ham': sum(r['label']==0 for r in v),
                           'spam': sum(r['label']==1 for r in v),
                           'truncated': sum(len(r['text'])>176 for r in v)} for k,v in splits.items()},
        'split_seed': 2026, 'sms_max_message_characters': 176,
    }
    (DATA/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    prepare()
