"""Pre-indexing of retrieval v2 / v3 on Kaggle: one kernel run scores every distinct (instruction, queries) search spec of a
set of companies over the whole news pool with the v2 hybrid (e5 + qwen + BM25, RRF) and the 30 checklist leaves with e5;
the result is loaded back as a `Frozen` search that answers `scores` from the file, so everything else (generators, gates,
ordering, evaluation) runs on CPU in seconds. Only query texts, news ids and event membership leave the machine: no
company ids, no facts. The kernel body is hybrid.py itself, so the ranking is the v2 one.
python -X utf8 stage3_news_search/retrieval_v3/precompute.py push --companies U_106783 … [--account kirpich682]
python -X utf8 stage3_news_search/retrieval_v3/precompute.py fetch   # kernel output -> data/retrieval_v2/precomputed.npz
Use in code: search = precompute.load(news, events); v2.hybrid_search = lambda *_: search"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import DATA
from stage3_news_search.retrieval_v2 import hybrid
from stage3_news_search.retrieval_v2.hybrid import HybridSearch
from stage3_news_search.retrieval_v3 import channels

HERE = Path(__file__).parent
NEWS_DATASET = 'maxamsecret/alfa-hack-ru-business-news'  # news.parquet = the corpus the local pool was built from
OUTFILE = DATA / 'retrieval_v2' / 'precomputed.npz'
CREDS = Path(os.path.expanduser('~')) / '.kaggle'
KERNEL_MAIN = '''
import glob, json
import numpy as np, pandas as pd
CFG = json.loads(r\'\'\'__CONFIG__\'\'\')
news = pd.read_parquet(glob.glob('/kaggle/input/**/news.parquet', recursive=True)[0])
events = pd.DataFrame(CFG['events'], columns=['event', 'news_ids'])
search = HybridSearch(news, events)
specs = [(s['instruction'], s['queries']) for s in CFG['specs']]
print(len(search.news), 'news in events;', len(specs), 'specs', flush=True)
S = search.scores(specs).astype(np.float32)
leaf = (search.docs['e5-large-instruct'] @ search.encoder('e5-large-instruct', CFG['leaf_queries']).T).astype(np.float32)
np.savez('/kaggle/working/precomputed.npz', news_id=search.news['id'].astype(str).to_numpy(), event=search.event,
         scores=S, leaf=leaf, spec_keys=np.array([json.dumps([i, q], ensure_ascii=False) for i, q in specs]))
print('done', S.shape, leaf.shape, flush=True)
'''


def spec_key(instruction: str, queries: list[str]) -> str:
    return json.dumps([instruction, list(queries)], ensure_ascii=False)


class Frozen(HybridSearch):
    """A HybridSearch without encoders: news / events as the real one builds them; `scores` from the recorded file or, with
    path=None, a recorder that returns zeros and collects the specs it was asked for (pass 1 of `push`)."""

    def __init__(self, news: pd.DataFrame, events: pd.DataFrame, path: Path | None = None):
        of = {str(i): int(e) for e, ids in zip(events.event, events.news_ids) for i in ids}
        event = np.array([of.get(str(i), -1) for i in news['id']])
        keep = event >= 0
        self.news, self.event = news[keep].reset_index(drop=True), event[keep]
        self.events = events.set_index('event')
        self.fingerprint = 'frozen'
        self.asked: dict[str, tuple[str, list[str]]] = {}
        self.table = None
        self.leaf = np.zeros((len(self.news), len(channels.LEAVES)), np.float32)  # recorder: dummy leaf scores
        if path is not None:
            z = np.load(path, allow_pickle=True)  # our own file: string arrays are object dtype
            assert list(z['news_id']) == self.news['id'].astype(str).tolist(), 'precomputed file belongs to another pool'
            assert (z['event'] == self.event).all(), 'precomputed file has another event membership'
            self.table = {k: j for j, k in enumerate(z['spec_keys'].tolist())}
            self.S, self.leaf = z['scores'], z['leaf']

    def scores(self, specs):
        keys = [spec_key(i, q) for i, q in specs]
        if self.table is None:
            self.asked |= {k: s for k, s in zip(keys, specs)}
            return np.zeros((len(self.news), len(specs)))
        missing = [k for k in keys if k not in self.table]
        assert not missing, f'{len(missing)} specs were not precomputed, e.g. {missing[0][:120]}'
        return self.S[:, [self.table[k] for k in keys]].astype(np.float64)


def load(news: pd.DataFrame, events: pd.DataFrame, path: Path = OUTFILE) -> Frozen:
    return Frozen(news, events, path)


def kernel_source(config: dict) -> str:
    src = Path(hybrid.__file__).read_text(encoding='utf-8')
    src = src.split("\nif __name__ == '__main__':")[0]
    for line in ("sys.path.insert(0, str(Path(__file__).resolve().parents[2]))", 'from common.data import DATA'):
        assert line in src, line
        src = src.replace(line, '')
    src = src.replace("CACHE = DATA / 'retrieval_v2'", "CACHE = Path('/kaggle/temp/retrieval_v2')")
    pip = "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'sentence-transformers>=5', 'bm25s', 'pymorphy3', 'pymorphy3-dicts-ru'], check=False)\n"
    return pip + src + KERNEL_MAIN.replace('__CONFIG__', json.dumps(config, ensure_ascii=False))


def push(companies: list[str], account: str, dry: bool) -> None:
    from stage2_company_facts.deterministic_facts.facts import build
    from stage3_news_search.retrieval_v2 import candidates as v2
    from stage3_news_search.retrieval_v3 import candidates as v3
    news, events = pd.read_parquet(DATA / 'news' / 'news.parquet'), pd.read_parquet(DATA / 'news' / 'events.parquet')
    rec = Frozen(news, events)
    v2.hybrid_search = lambda *_: rec  # pass 1: run the real code against the recorder
    facts = build(companies)
    for u in companies:
        v3.candidates(u, facts[u], events=events, news=news, frozen_factory=True)
    config = {'specs': [{'instruction': i, 'queries': q} for i, q in rec.asked.values()],
              'leaf_queries': channels.leaf_queries(),
              'events': [[int(e), [str(i) for i in ids]] for e, ids in zip(events.event, events.news_ids)]}
    print(f'{len(config["specs"])} distinct specs for {len(companies)} companies, {len(events)} events', flush=True)
    d = HERE / 'kernels' / 'v3_scores'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'precompute_kernel.py').write_text(kernel_source(config), encoding='utf-8')
    (d / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'{account}/alfa-v3-scores', 'title': 'alfa-v3-scores', 'code_file': 'precompute_kernel.py', 'language': 'python',
        'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
        'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [NEWS_DATASET], 'competition_sources': [], 'kernel_sources': [],
        'model_sources': []}, indent=1), encoding='utf-8')
    print('kernel ->', d)
    if not dry:
        subprocess.run(['kaggle', 'kernels', 'push', '-p', str(d)], check=False,
                       env={**os.environ, 'KAGGLE_CONFIG_DIR': str(CREDS / f'kaggle_{account}' / '.kaggle')})


def fetch(account: str) -> None:
    out = OUTFILE.parent
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(['kaggle', 'kernels', 'output', f'{account}/alfa-v3-scores', '-p', str(out), '--force'], check=True,
                   env={**os.environ, 'KAGGLE_CONFIG_DIR': str(CREDS / f'kaggle_{account}' / '.kaggle'), 'PYTHONIOENCODING': 'utf-8'})
    print(OUTFILE, OUTFILE.exists())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['push', 'fetch'])
    ap.add_argument('--companies', nargs='+', default=['U_106783', 'U_9250', 'U_150736', 'U_212593'])
    ap.add_argument('--account', default='kirpich682')
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    push(a.companies, a.account, a.dry) if a.cmd == 'push' else fetch(a.account)
