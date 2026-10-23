"""Migrate saved artifacts from row-number news ids (old build.py) to the stable url-hash ids (build.stable_id).

Two id spaces are rewritten:
  news  id: old = row number in the old news.parquet -> new = stable_id(url). Exact, needs only the old news.parquet.
  event id: old = row in the old events.parquet (clusters of news; positional, changes with every corpus rebuild)
            -> the new event holding most of the old event's members. Needs --new-events (events.py run on the new corpus).

  python migrate_ids.py [--new-events data/news/events_new.parquet]            # dry run: report only, writes nothing
  python migrate_ids.py --new-events ... --apply [--drop-unmapped]             # back up to data/_id_migration_backup, rewrite

Not migrated: experiments/retrieval/labels_{cheap,agent}.jsonl - v1 labels keyed by an even older events numbering
(events 8037-8187 do not exist in the current events.parquet, so their ids are already misaligned); superseded by judge v2.
Regenerate: data/news/{news,events}.parquet, data/llm_runs/* (experiments/llm_runs/prepare.py),
data/llm_cache + data/logs (hash keys / history), docs that quote ids in prose.
"""
import argparse, importlib, json, shutil, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


class Unmapped(Exception):
    pass


class Mapper:
    def __init__(self, news_map, event_map, old_members):
        self.nmap, self.emap, self.members = news_map, event_map, old_members
        self.stats = Counter()

    def _one(self, space, m, x):
        try:
            y = m[int(x)]
        except (KeyError, ValueError):
            self.stats[f'unmapped_{space}'] += 1
            raise Unmapped
        self.stats[f'{space}_rewritten'] += 1
        return type(x)(y) if isinstance(x, (int, str)) else y

    def N(self, x):
        return self._one('news', self.nmap, x)

    def E(self, x):
        return self._one('event', self.emap, x)

    def Ns(self, xs):
        return self._many(self.N, xs)

    def Es(self, xs):
        return self._many(self.E, xs)

    def _many(self, f, xs):
        out = []
        for x in xs:
            try:
                out.append(f(x))
            except Unmapped:
                pass  # counted in stats; the entry is dropped (apply refuses unless --drop-unmapped)
        return out

    def pair_ok(self, e, n):
        """Old-space sanity check: the news of an (event, news) pair must be a member of that event."""
        ok = self.members is None or int(n) in self.members.get(int(e), ())
        self.stats['pair_checked'] += 1
        self.stats['pair_mismatch'] += not ok


def part(key, i, f):
    p = str(key).split('|')
    p[i] = str(f(p[i]))
    return '|'.join(p)


# ---- per-format rules: each returns the migrated object, the rule decides which ids are news / event ----
def dict_of_event_lists(d, m):
    return {k: m.Es(v) for k, v in d.items()}


def _agent(rec, m):
    v = rec['value']
    rec['key'] = part(rec['key'], 1, m.E)  # 'role|event'
    m.pair_ok(v['event'], v['news_id'])
    v['event'], v['news_id'] = m.E(v['event']), m.N(v['news_id'])
    d = v.get('decision') or {}
    if 'related_news_ids' in d:
        d['related_news_ids'] = m.Ns(d['related_news_ids'])
    return rec


def _labels_v2(rec, m):
    v = rec['value']
    m.pair_ok(v['event'], v['news_id'])
    rec['key'] = part(rec['key'], 0, m.E)  # 'event|input|refusal'
    v['event'], v['news_id'] = m.E(v['event']), m.N(v['news_id'])
    return rec


def _labels_pack(rec, m):
    rec['key'] = part(rec['key'], 1, m.E)  # 'pack|event'
    rec['value']['event'] = m.E(rec['value']['event'])
    return rec


def _event_key_news_value(rec, m):  # judge v1 labels, glm scores: key = event, value.news_id = its representative news
    v = rec['value']
    m.pair_ok(rec['key'], v['news_id'])
    rec['key'], v['news_id'] = str(m.E(rec['key'])), m.N(v['news_id'])
    return rec


def _useful_negatives(d, m):
    items = []
    for x in d['items']:
        try:
            items.append(dict(x, event=m.E(x['event'])))
        except Unmapped:
            pass
    d['items'] = items
    return d


def _real_pairs(d, m):
    d['pairs'] = {u: m.Ns(ids) for u, ids in d['pairs'].items()}
    return d


def _events_json(d, m):
    return {k: m.Es(v) for k, v in d.items()}


def _generator_tops(d, m):
    d['tops'] = dict_of_event_lists(d['tops'], m)
    return d


def _standard_sets(d, m):
    for v in d.values():
        v['top'] = m.Es(v['top'])
    return d


OUT = 'experiments'
RULES = [
    # (glob, format, function) ; json: obj -> obj ; jsonl: record -> record ; parquet/csv/pkl: (column or index, space)
    (f'{OUT}/generator_factory/out/factory_*_todo.json', 'json', lambda d, m: m.Es(d)),
    (f'{OUT}/generator_factory/out/factory_*_unlabelled.json', 'json', lambda d, m: m.Es(d)),
    (f'{OUT}/generator_factory/out/judge_events_*.json', 'json', lambda d, m: m.Es(d)),
    (f'{OUT}/generator_factory/out/judge_sets_*.json', 'json', dict_of_event_lists),
    (f'{OUT}/generator_factory/out/factory_*_tops*.json', 'json', dict_of_event_lists),
    (f'{OUT}/judge_v2/out/new_gen_unlabelled.json', 'json', lambda d, m: m.Es(d)),
    (f'{OUT}/judge_v2/out/generator_sets.json', 'json', dict_of_event_lists),
    (f'{OUT}/judge_v2/out/generator_tops.json', 'json', _generator_tops),
    (f'{OUT}/judge_v2/useful_negatives.json', 'json', _useful_negatives),
    (f'{OUT}/candidate_sets/out/standard_*.json', 'json', _standard_sets),
    (f'{OUT}/news_channels/out/events.json', 'json', _events_json),
    (f'{OUT}/news_channels/out/random_events.json', 'json', lambda d, m: m.Es(d)),
    ('stage3_news_search/golden_set/real_pairs.json', 'json', _real_pairs),
    (f'{OUT}/judge_v2/out/agent_*.jsonl', 'jsonl', _agent),
    (f'{OUT}/judge_v2/out/labels_v2_*.jsonl', 'jsonl', _labels_v2),
    (f'{OUT}/judge_v2/out/labels_pack_*.jsonl', 'jsonl', _labels_pack),
    (f'{OUT}/judge_v2/out/labels_U_*.jsonl', 'jsonl', _event_key_news_value),
    (f'{OUT}/news_channels/out/glm_scores.jsonl', 'jsonl', _event_key_news_value),
    (f'{OUT}/judge_v2/out/summaries*.jsonl', 'jsonl', lambda r, m: dict(r, key=str(m.N(r['key'])))),  # key = event's news
    (f'{OUT}/news_channels/out/channels_news.parquet', 'parquet', ('news_id', 'news')),
    (f'{OUT}/news_channels/out/channels_events.parquet', 'parquet', ('event', 'event')),
    (f'{OUT}/retrieval/out/hypothesis_hooks_*.csv', 'csv', ('event', 'event')),
    (f'{OUT}/judge_v2/out/single.pkl', 'pkl', ('event', 'event')),
    (f'{OUT}/judge_v2/out/consensus.pkl', 'pkl', ('index:event', 'event')),
    # arrays are aligned with their id array, so rewriting the id array keeps rows valid (row order is NOT the corpus order)
    (f'{OUT}/news_channels/out/rank/*.npz', 'npz', ('keys', 'event')),
    (f'{OUT}/news_channels/out/rank_all/*.npz', 'npz', ('keys', 'news')),
    (f'{OUT}/retrieval/out/*.npz', 'npz', ('doc_ids', 'news')),
    (f'{OUT}/retrieval/out/*_doc_ids.npy', 'npy', (None, 'news')),
    (f'{OUT}/retrieval/dl_*/*.npz', 'npz', ('doc_ids', 'news')),
    (f'{OUT}/retrieval/dl_*/*_doc_ids.npy', 'npy', (None, 'news')),
]


def style_of(text, obj):
    """json.dumps arguments that reproduce the file byte for byte (so only ids change); None if none do."""
    for ascii_ in (True, False):
        for indent in (None, 1, 2):
            s = json.dumps(obj, ensure_ascii=ascii_, indent=indent)
            if s == text or s + '\n' == text:
                return dict(ensure_ascii=ascii_, indent=indent), text.endswith('\n')
    return None, text.endswith('\n')


def map_series(s, f, m):
    """Map an id Series element-wise keeping its dtype; unmapped rows are dropped (returned mask)."""
    keep, out = [], []
    for x in s:
        try:
            out.append(f(x)); keep.append(True)
        except Unmapped:
            out.append(x); keep.append(False)
    return pd.Series(out, index=s.index, dtype=s.dtype if s.dtype != object else object), np.array(keep)


def migrate_file(path, fmt, spec, m, out_path=None):
    """Returns (new_content_writer, info). Nothing is written unless writer() is called."""
    f = {'news': m.N, 'event': m.E}
    before = Counter(m.stats)
    if fmt == 'json':
        text = path.read_text(encoding='utf-8'); obj = json.loads(text)
        style, nl = style_of(text, obj)
        new = spec(obj, m)
        dump = lambda: json.dumps(new, **(style or dict(ensure_ascii=False, indent=1))) + ('\n' if nl else '')
        writer = lambda: path.write_text(dump(), encoding='utf-8')
        exact = style is not None
    elif fmt == 'jsonl':
        lines = [l for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        exact = all(json.dumps(json.loads(l), ensure_ascii=False) == l for l in lines)
        recs = []
        for l in lines:
            try:
                recs.append(spec(json.loads(l), m))
            except Unmapped:
                pass
        writer = lambda: path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in recs), encoding='utf-8')
    elif fmt in ('parquet', 'csv', 'pkl'):
        col, space = spec
        df = {'parquet': pd.read_parquet, 'csv': pd.read_csv, 'pkl': pd.read_pickle}[fmt](path)
        if col.startswith('index:'):
            idx = df.index.map(lambda x: x)  # DataFrame indexed by event
            new_idx, keep = map_series(pd.Series(idx), f[space], m)
            df = df[keep]; df.index = pd.Index(new_idx[keep].values, name=col[6:])
        else:
            df[col], keep = map_series(df[col], f[space], m)
            df = df[keep]
        exact = True
        writer = lambda: {'parquet': lambda: df.to_parquet(path, index=False), 'csv': lambda: df.to_csv(path, index=False),
                          'pkl': lambda: df.to_pickle(path)}[fmt]()
    else:  # npz / npy
        col, space = spec
        if fmt == 'npz':
            z = dict(np.load(path, allow_pickle=False))
            if col not in z:
                return (lambda: None), {'no_id_array': 1, 'byte_exact_format': True}
            ids = z[col]
        else:
            z = None; ids = np.load(path, allow_pickle=False)
        new = []
        for x in ids:
            x = x.item()
            try:
                new.append(f[space](x) if str(x).lstrip('-').isdigit() else x)  # synthetic needle docs keep their own ids
            except Unmapped:
                new.append(x)
        new = np.array(new)  # new dtype: stable ids are wider than the old '<U4' keys
        exact = True
        if fmt == 'npz':
            z[col] = new; writer = lambda: np.savez(path, **z)
        else:
            writer = lambda: np.save(path, new)
    info = {k: m.stats[k] - before[k] for k in m.stats if m.stats[k] != before[k]} | {'byte_exact_format': exact}
    return writer, info


def build_maps(root, new_events_path):
    sys.path.insert(0, str(HERE))
    if 'def stable_id' not in (HERE / 'build.py').read_text(encoding='utf-8'):
        sys.exit('build.py here is the old collector (no stable_id): run this after the new build.py is merged')
    stable_id = importlib.import_module('build').stable_id
    old = pd.read_parquet(root / 'data/news/news.parquet', columns=['id', 'url'])
    assert old.id.is_unique and old.url.is_unique
    old['new'] = old.url.map(stable_id)
    assert old.new.is_unique, 'stable_id collision inside the old corpus'
    nmap = dict(zip(old.id.astype(int), old.new.astype(int)))
    oev = pd.read_parquet(root / 'data/news/events.parquet', columns=['event', 'news_ids'])
    members = {int(e): [int(i) for i in ids] for e, ids in zip(oev.event, oev.news_ids)}
    emap, report = {e: e for e in members}, {}  # identity without --new-events: only the old-space checks are meaningful
    if new_events_path:
        nev = pd.read_parquet(new_events_path, columns=['event', 'news_ids'])
        where = {int(i): int(e) for e, ids in zip(nev.event, nev.news_ids) for i in ids}
        emap, classes = {}, Counter()
        for e, ids in members.items():
            hit = Counter(where[nmap[i]] for i in ids if nmap[i] in where)
            if not hit:
                classes['lost (no member in new corpus)'] += 1; continue
            best, n = hit.most_common(1)[0]
            emap[e] = best
            classes['whole' if n == len(ids) and len(hit) == 1 else 'split (best holds >=50%)' if n * 2 >= len(ids) else 'split (best holds <50%)'] += 1
        report = dict(classes)
    return old, nmap, emap, members, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(HERE.parents[1]), help='repo root')
    ap.add_argument('--new-events', help='events.parquet built from the new corpus (event id mapping)')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--drop-unmapped', action='store_true', help='with --apply: allow dropping entries whose event/news is gone')
    a = ap.parse_args()
    root = Path(a.root)
    old, nmap, emap, members, ev_report = build_maps(root, a.new_events)
    print(f'news map: {len(nmap)} ids; identity for {sum(k == v for k, v in nmap.items())}')
    print('event map:', ev_report or 'IDENTITY (no --new-events): event ids unchanged, old-space checks only')
    marker = root / 'data/_id_migration_done.json'
    if a.apply and (marker.exists() or not a.new_events):
        sys.exit('refusing: already applied' if marker.exists() else '--apply needs --new-events')

    plan, totals = [], Counter()
    for pattern, fmt, spec in RULES:
        for path in sorted(root.glob(pattern)):
            m = Mapper(nmap, emap, members)
            writer, info = migrate_file(path, fmt, spec, m)
            plan.append((path, writer, info))
            totals.update({k: v for k, v in info.items() if isinstance(v, int) and not isinstance(v, bool)})
            flags = ' '.join(f'{k}={v}' for k, v in info.items() if k != 'byte_exact_format') + ('' if info['byte_exact_format'] else ' FORMAT-CHANGES')
            print(f'  {str(path.relative_to(root)):75s} {flags}')
    print('totals:', dict(totals))
    bad = totals['pair_mismatch']
    if bad:
        print(f'WARNING: {bad} (event, news) pairs where the news is not a member of the event in the OLD tables')
    if not a.apply:
        print('dry run: nothing written')
        return
    if (totals['unmapped_news'] or totals['unmapped_event']) and not a.drop_unmapped:
        sys.exit('refusing: unmapped ids present, rerun with --drop-unmapped to drop those entries')
    backup = root / 'data/_id_migration_backup'
    for path, _, _ in plan:
        dst = backup / path.relative_to(root); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dst)
    for _, writer, _ in plan:
        writer()
    pd.DataFrame({'old_id': list(nmap), 'new_id': list(nmap.values())}).to_parquet(root / 'data/news/news_id_map.parquet', index=False)
    marker.write_text(json.dumps({'files': len(plan), 'totals': dict(totals), 'events': ev_report}))
    print(f'applied to {len(plan)} files; originals in {backup}')


if __name__ == '__main__':
    main()
