"""Build and push the ranker kernels (kaggle_rank.py): the 398 judge-labelled events of U_106783 + the 400 random
events GLM scores, each as title + first TEXT_CHARS of its first report (no company data leaves the machine).
--all: every news of data/news/news.parquet (key = news id), `item` variant only (the ensemble's), kernels <name>-all.
python -X utf8 experiments/news_channels/push.py [--all] [--only giga small] [--dry]
Fetch: kaggle kernels output <account>/alfa-rank-<name>[-all] -p experiments/news_channels/out/rank[_all] (KAGGLE_CONFIG_DIR per account)."""
import argparse, json, os, shutil, subprocess
from pathlib import Path

import pandas as pd

from channels import DATA, HERE, ITEMS, TEXT_CHARS, labelled_events, random_events
from judge import event_news

CREDS = Path(os.path.expanduser('~')) / '.kaggle'
VARIANTS = {
    'item': '{item}',
    'question': 'Может ли эта новость через канал «{item}» заметно изменить выручку, затраты, доступ к рынку или к финансированию конкретного бизнеса?',
}
INSTRUCTION = ('Найди новости о событиях, которые через указанный канал влияния могут заметно изменить выручку, затраты, '
               'доступ к рынку или к финансированию конкретных компаний')
MODELS = [
    {'slug': 'bge-reranker-v2-m3', 'kind': 'cross', 'name': 'BAAI/bge-reranker-v2-m3', 'bs': 64},
    {'slug': 'ru-reranker-edge-150m', 'kind': 'cross', 'name': 'sshalimov04/ru-reranker-edge-150m', 'bs': 128},
    {'slug': 'e5-large-instruct', 'kind': 'dense', 'name': 'intfloat/multilingual-e5-large-instruct', 'bs': 32},
    {'slug': 'giga-emb-480m', 'kind': 'dense', 'name': 'ai-sage/Giga-Embeddings-instruct-480M-0826', 'dtype': 'float32', 'bs': 16},
    {'slug': 'giga-emb-10b', 'kind': 'hf', 'name': 'ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826', 'bs': 8},
]
KERNELS = {  # name: (account, [(CUDA_VISIBLE_DEVICES, [slugs in order])])
    'giga': ('kirpich682', [('0,1', ['giga-emb-10b'])]),
    'small': ('alexandrzolotukhin', [('0', ['bge-reranker-v2-m3']), ('1', ['ru-reranker-edge-150m', 'e5-large-instruct', 'giga-emb-480m'])]),
}
PIP = ['sentence-transformers==5.1.1', 'transformers>=4.51,<5', 'accelerate']


def news():
    """The kernel rebuilds the text from the dataset's news.parquet (same file as data/news); only ids go in the config."""
    events = list(dict.fromkeys(sorted(labelled_events()) + random_events()))
    texts = event_news(events)
    return [{'key': e, 'id': texts[e]['id']} for e in events]


def all_news():
    return [{'key': i, 'id': i} for i in pd.read_parquet(DATA / 'news' / 'news.parquet', columns=['id'])['id'].astype(str)]


def build(name, account, groups, items, variants, item_texts=ITEMS):
    d = HERE / 'kernels' / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    slugs = {s for _, ss in groups for s in ss}
    cfg = {'news': items, 'text_chars': TEXT_CHARS, 'items': item_texts, 'variants': variants, 'variant_templates': VARIANTS, 'instruction': INSTRUCTION,
           'models': [m for m in MODELS if m['slug'] in slugs], 'gpu_groups': groups, 'pip': PIP}
    body = (HERE / 'kaggle_rank.py').read_text(encoding='utf-8').replace('__CONFIG__', json.dumps(cfg, ensure_ascii=False))
    (d / 'rank.py').write_text(body, encoding='utf-8')
    (d / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'{account}/alfa-rank-{name}', 'title': f'alfa-rank-{name}', 'code_file': 'rank.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': ['maxamsecret/alfa-hack-ru-business-news'], 'competition_sources': [], 'kernel_sources': [],
        'model_sources': []}, indent=1), encoding='utf-8')
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', default=list(KERNELS))
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--all', action='store_true', help='whole corpus, item variant only')
    a = ap.parse_args()
    items = all_news() if a.all else news()
    print(len(items), 'news')
    for name in a.only:
        account, groups = KERNELS[name]
        d = build(name + '-all' * a.all, account, groups, items, ['item'] if a.all else list(VARIANTS))
        print(name, '->', d)
        if not a.dry:
            env = {**os.environ, 'KAGGLE_CONFIG_DIR': str(CREDS / f'kaggle_{account}' / '.kaggle')}
            subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], env=env, check=False)
