"""Build and push the embedding kernels: one Kaggle kernel per account, models spread over GPU lanes.
python -X utf8 experiments/retrieval/push.py [--only med qwen8b giga] [--dry]
Queries are the unique generator texts of the eval companies (queries.json, no company ids leave the machine)."""
import argparse, json, os, shutil, subprocess
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CREDS = Path(os.path.expanduser('~')) / '.kaggle'
DATASET = 'maxamsecret/alfa-hack-ru-business-news'

INSTRUCTIONS = {  # task descriptions tried on instruct encoders (Instruct: …\nQuery: …)
    'web': 'Given a web search query, retrieve relevant passages that answer the query',
    'en_generic': "Given a description of how a company's business is exposed, retrieve news articles about events that could affect this business",
    'ru_generic': 'Дано описание того, с чем связан бизнес компании. Найди новости о событиях, которые могут повлиять на этот бизнес',
    'ru_insight': 'Найди новости, из которых банк может сделать бизнес-инсайт для компании: события, которые через указанный фактор создают для неё риски или возможности',
}
REASON_INSTRUCTIONS = {
    'own_industry': 'Найди новости о регулировании, ценах, спросе и конкуренции в этой отрасли',
    'supplier_industry': 'Найди новости о событиях, сильно влияющих на эту отрасль поставщиков компании: цены, дефицит, регулирование, сбои',
    'customer_industry': 'Найди новости о событиях, сильно влияющих на эту отрасль покупателей компании: спрос, финансы, регулирование',
    'second_hop': 'Найди новости о событиях в этой отрасли, которые могут пройти по цепочке поставок до компании',
    'group_industry': 'Найди новости о событиях в этой отрасли, важных для группы компаний',
    'bank_product': 'Найди новости об условиях и изменениях по этому банковскому продукту или фактору для бизнеса',
    'macro': 'Найди новости о макроэкономическом факторе, влияющем на бизнес компаний',
    'trade_country': 'Найди новости о торговле, пошлинах, санкциях и логистике с этой страной',
    'region': 'Найди новости об экономике и бизнесе этого региона',
    'hypothesis': 'Найди новости, важные для бизнеса с такой характеристикой',
}
MODELS = [
    {'slug': 'frida', 'kind': 'dense', 'name': 'ai-forever/FRIDA', 'family': 'prefix', 'qprefix': 'search_query: ', 'dprefix': 'search_document: ', 'bs': 16},
    {'slug': 'user-bge-m3', 'kind': 'dense', 'name': 'deepvk/USER-bge-m3', 'family': 'prefix', 'bs': 16},
    {'slug': 'e5-large-instruct', 'kind': 'dense', 'name': 'intfloat/multilingual-e5-large-instruct', 'family': 'instruct', 'bs': 16},
    {'slug': 'qwen3-emb-0.6b', 'kind': 'dense', 'name': 'Qwen/Qwen3-Embedding-0.6B', 'family': 'instruct_nospace', 'padding_side': 'left', 'bs': 16},
    {'slug': 'bge-m3', 'kind': 'bge_m3', 'name': 'BAAI/bge-m3', 'family': 'prefix', 'bs': 16},
    {'slug': 'splade-os-multi', 'kind': 'sparse', 'name': 'opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1', 'family': 'prefix', 'bs': 4},
    {'slug': 'qwen3-emb-8b', 'kind': 'hf', 'name': 'Qwen/Qwen3-Embedding-8B', 'family': 'instruct_nospace', 'padding_side': 'left', 'pooling': 'last', 'bs': 8},
    {'slug': 'giga-emb-10b', 'kind': 'hf', 'name': 'ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826', 'family': 'instruct', 'pooling': 'mean', 'bs': 8},
]
KERNELS = {  # name: (account, [(CUDA_VISIBLE_DEVICES, [slugs in order])])
    'med': ('alexandrzolotukhin', [('0', ['frida', 'e5-large-instruct', 'qwen3-emb-0.6b']), ('1', ['user-bge-m3', 'bge-m3', 'splade-os-multi'])]),
    'qwen8b': ('annkrasnoshchekova', [('0,1', ['qwen3-emb-8b'])]),
    'giga': ('kirpich682', [('0,1', ['giga-emb-10b'])]),
}
PIP = ['sentence-transformers==5.1.1', 'transformers>=4.51,<5', 'accelerate', 'FlagEmbedding==1.3.5']


def config(slugs):
    qs = json.loads((HERE / 'queries.json').read_text(encoding='utf-8'))['queries']
    uniq = list({(q['reason'], q['query'], q['text']): None for q in qs.values()})
    needles = json.loads((ROOT / 'stage3_news_search/golden_set/synthetic_needles.json').read_text(encoding='utf-8'))
    nd = [{'id': f'needle|{u}|{n["kind"]}', 'title': n['title'], 'text': n['text']} for u, ns in needles['needles'].items() for n in ns]
    nd += [{'id': f'neg|{k}', 'title': n['title'], 'text': n['text']} for k, n in enumerate(needles['hard_negatives'])]
    (HERE / 'query_index.json').write_text(json.dumps([dict(zip(('reason', 'query', 'text'), u)) for u in uniq], ensure_ascii=False), encoding='utf-8')
    return {'news_path': f'/kaggle/input/{DATASET.split("/")[1]}/news.parquet', 'needles': nd,
            'queries': [dict(zip(('reason', 'query', 'text'), u)) for u in uniq], 'instructions': INSTRUCTIONS,
            'reason_instructions': REASON_INSTRUCTIONS, 'models': [m for m in MODELS if m['slug'] in slugs], 'pip': PIP}


def build(name, account, groups):
    d = HERE / 'kernels' / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    cfg = config({s for _, ss in groups for s in ss}) | {'gpu_groups': groups}
    body = (HERE / 'kaggle_embed.py').read_text(encoding='utf-8').replace('__CONFIG__', json.dumps(cfg, ensure_ascii=False))
    (d / 'embed.py').write_text(body, encoding='utf-8')
    (d / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'{account}/alfa-emb-{name}', 'title': f'alfa-emb-{name}', 'code_file': 'embed.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [DATASET], 'competition_sources': [], 'kernel_sources': [],
        'model_sources': []}, indent=1), encoding='utf-8')
    return d


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', default=list(KERNELS))
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for name in a.only:
        account, groups = KERNELS[name]
        d = build(name, account, groups)
        print(name, '->', d)
        if not a.dry:
            env = {**os.environ, 'KAGGLE_CONFIG_DIR': str(CREDS / f'kaggle_{account}' / '.kaggle')}
            subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], env=env, check=False)
