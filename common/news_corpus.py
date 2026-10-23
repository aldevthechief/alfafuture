"""Publish news snapshots and resolve persisted references without trusting recycled numeric ids."""
import hashlib
import json
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.search import corpus_fingerprint

ID_SCHEME = 'url-sha256-63-v1'
TRACKING = {'fbclid', 'gclid', 'yclid', 'ysclid', 'ref_src'}


class StaleNewsReferences(ValueError):
    pass


def normalize_url(url: str) -> str:
    parts = urlsplit(str(url).strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in TRACKING]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip('/') or '/',
                       urlencode(sorted(query)), ''))


def stable_id(url: str) -> int:
    return int.from_bytes(hashlib.sha256(normalize_url(url).encode()).digest()[:8], 'big') & ((1 << 63) - 1)


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem + '-', suffix=path.suffix, delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        write(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    _atomic(Path(path), lambda p: p.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str), encoding='utf-8'))


def archive_corpus(news: pd.DataFrame, folder: Path) -> Path:
    fingerprint = corpus_fingerprint(news)
    path = Path(folder) / 'versions' / fingerprint / 'news.parquet'
    if not path.exists():
        snapshot = news.copy()
        snapshot.attrs['corpus_fingerprint'] = fingerprint
        _atomic(path, lambda p: snapshot.to_parquet(p, index=False))
    return path


def article_fingerprint(row: dict) -> str:
    # ID is deliberately absent: a URL-identical article may get another ID after migration.
    def text(key):
        value = row.get(key)
        return '' if value is None or pd.isna(value) else str(value)
    values = [normalize_url(text('url')), *[text(k) for k in ('title', 'text', 'date', 'source')]]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def news_reference(row: dict) -> dict:
    return {'id': str(row['id']), 'url': normalize_url(row.get('url') or ''),
            'content_fingerprint': article_fingerprint(row),
            'title': row.get('title', ''), 'date': str(row.get('date') or ''), 'source': row.get('source', '')}


def referenced_news_ids(value) -> set[str]:
    ids = set()
    def visit(v):
        if isinstance(v, str) and v.startswith('news:'):
            ids.add(v[5:])
        elif isinstance(v, list):
            for x in v:
                visit(x)
        elif isinstance(v, dict):
            for key, x in v.items():
                if key == 'news_id' and x is not None:
                    ids.add(str(x))
                elif key in ('news_ids', 'related_news_ids', 'event_news_ids') and x is not None:
                    ids.update(str(i) for i in x)
                elif key == 'news' and isinstance(x, list):
                    ids.update(str(r['id']) for r in x if isinstance(r, dict) and 'id' in r)
                visit(x)
    visit(value)
    return ids


def unresolved_ids(result: dict) -> set[str]:
    """Only rejected diagnostics may keep invented citations, clearly separate from bound sources."""
    if 'insight' in result or result.get('status') == 'ready':
        return set()
    if result.get('status') not in ('audit_rejected', 'review_rejected', 'model_error') and result.get('verdict', {}).get('status') != 'reject':
        return set()
    return {str(i) for i in result.get('unresolved_news_ids', [])} - {str(result.get('news_id'))}


def bind_result(result: dict, news: pd.DataFrame, *, allow_unresolved: bool = False) -> dict:
    fingerprint = corpus_fingerprint(news)
    if result.get('corpus_fingerprint') and result['corpus_fingerprint'] != fingerprint:
        raise StaleNewsReferences('Cannot rebind a historical result to another corpus snapshot')
    by_id = {str(r['id']): r for r in news.to_dict('records')}
    ids = referenced_news_ids(result)
    missing = ids - by_id.keys()
    if missing and allow_unresolved and str(result.get('news_id')) not in missing:
        result = dict(result)
        if 'insight' in result:
            result['draft'] = result.pop('insight')
        result.update(status='audit_rejected', unresolved_news_ids=sorted(missing))
        result['verdict'] = dict(result.get('verdict') or {}, status='reject', reasons=['Unknown news citations in model output'])
        result['audit'] = [*result.get('audit', []), {'section': 'references', 'code': 'unknown_ref',
                                                    'refs': [f'news:{i}' for i in sorted(missing)], 'text': 'Unknown news citations'}]
    if missing - unresolved_ids(result):
        raise StaleNewsReferences(f'Cannot save result: {len(missing)} news references are absent from its source snapshot')
    return dict(result, corpus_fingerprint=fingerprint,
                news_references={f'news:{i}': news_reference(by_id[i]) for i in sorted(ids - missing)})


def _url_index(news: pd.DataFrame) -> dict[str, list[dict]]:
    out = {}
    for row in news.to_dict('records'):
        if row.get('url'):
            out.setdefault(normalize_url(row['url']), []).append(row)
    return out


def resolve_reference(ref: dict, news: pd.DataFrame | None = None, *, index: dict | None = None) -> dict:
    index = _url_index(news) if index is None else index
    if not ref.get('url') or not ref.get('content_fingerprint'):
        return {'status': 'unverified', 'original_id': str(ref.get('id', ''))}
    matches = index.get(normalize_url(ref['url']), [])
    if len(matches) != 1:
        return {'status': 'missing' if not matches else 'ambiguous', 'original_id': str(ref['id']), 'url': ref['url']}
    row = matches[0]
    if article_fingerprint(row) != ref['content_fingerprint']:
        status = 'changed'
    else:
        status = 'current' if str(row['id']) == str(ref['id']) else 'remapped'
    return {'status': status, 'original_id': str(ref['id']), 'current_id': str(row['id']), 'url': ref['url']}


def result_references(result: dict, news: pd.DataFrame) -> dict:
    if not isinstance(result, dict):
        return {'ok': False, 'status': 'unverified', 'reason': 'Invalid saved result'}
    refs = result.get('news_references') or {}
    required = referenced_news_ids({k: v for k, v in result.items() if k not in (
        'news_references', 'resolved_news_references')})
    required -= unresolved_ids(result)
    if not result.get('corpus_fingerprint') or any(f'news:{i}' not in refs for i in required):
        return {'ok': False, 'status': 'unverified', 'reason': 'No verified source snapshot; numeric IDs cannot be trusted'}
    index = _url_index(news)
    resolved = {key: resolve_reference(ref, index=index) for key, ref in refs.items()}
    ok = all(r['status'] in ('current', 'remapped') for r in resolved.values())
    return {'ok': ok, 'status': 'current' if ok else 'stale', 'references': resolved}


def save_results(path: Path, results: list[dict], news: pd.DataFrame, news_folder: Path) -> None:
    bound = [bind_result(r, news) for r in results]
    archive_corpus(news, news_folder)
    write_json(path, bound)


def load_results(path: Path, news: pd.DataFrame) -> list[dict]:
    """Historical claim IDs stay immutable; resolved references contain validated IDs of the current corpus."""
    results = json.loads(Path(path).read_text(encoding='utf-8'))
    out = []
    for r in results:
        state = result_references(r, news)
        if not state['ok']:
            raise StaleNewsReferences(f'{Path(path).name}: {state["status"]} news references; use its archived snapshot or regenerate')
        out.append(dict(r, resolved_news_references=state['references']))
    return out


def load_history(path: Path, news_folder: Path) -> list[dict]:
    """Read the actual old sources; do not reinterpret a historical claim using an updated article."""
    results = json.loads(Path(path).read_text(encoding='utf-8'))
    out = []
    for r in results:
        fingerprint = r.get('corpus_fingerprint', '')
        if len(fingerprint) != 64 or any(c not in '0123456789abcdef' for c in fingerprint):
            raise StaleNewsReferences('Historical result has no verified corpus snapshot')
        source = Path(news_folder) / 'versions' / fingerprint / 'news.parquet'
        if not source.exists():
            raise StaleNewsReferences('Original corpus snapshot is unavailable')
        news = pd.read_parquet(source)
        if corpus_fingerprint(news) != fingerprint or not result_references(r, news)['ok']:
            raise StaleNewsReferences('Historical source snapshot does not match the saved references')
        ids = referenced_news_ids(r)
        out.append(dict(r, historical_news=[row for row in news.to_dict('records') if str(row['id']) in ids]))
    return out


def bind_pairs(path: Path, news: pd.DataFrame) -> Path:
    """Explicit bootstrap while the ORIGINAL corpus is still available; never infer provenance from new IDs."""
    path = Path(path)
    raw = path.read_bytes()
    pairs = json.loads(raw)['pairs']
    ids = {str(i) for values in pairs.values() for i in values}
    by_id = {str(r['id']): r for r in news.to_dict('records')}
    if ids - by_id.keys():
        raise StaleNewsReferences('Cannot bind pairs: original news IDs are missing')
    target = path.with_suffix('.sources.json')
    metadata = {'file_sha256': hashlib.sha256(raw).hexdigest(), 'corpus_fingerprint': corpus_fingerprint(news),
                'references': {i: news_reference(by_id[i]) for i in sorted(ids)}}
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != metadata:
        raise StaleNewsReferences('Pairs already have another source binding; do not overwrite their provenance')
    write_json(target, metadata)
    return target


def load_pairs(path: Path, news: pd.DataFrame) -> dict:
    path = Path(path)
    source = path.with_suffix('.sources.json')
    if not source.exists():
        raise StaleNewsReferences('Pairs have no source binding; bind them against their original corpus first')
    meta = json.loads(source.read_text(encoding='utf-8'))
    if meta['file_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise StaleNewsReferences('Pairs changed since their source binding was created')
    index = _url_index(news)
    resolved = {i: resolve_reference(ref, index=index) for i, ref in meta['references'].items()}
    if any(r['status'] not in ('current', 'remapped') for r in resolved.values()):
        raise StaleNewsReferences('Pairs contain removed or changed articles; revise the labels against the current corpus')
    pairs = json.loads(path.read_text(encoding='utf-8'))['pairs']
    return {u: [int(resolved[str(i)]['current_id']) for i in ids] for u, ids in pairs.items()}


def filtered_news(news: pd.DataFrame) -> pd.DataFrame:
    return news[~news.title.str.contains('пошлина на ввоз по коду', na=False)].reset_index(drop=True)


def publish_news(news: pd.DataFrame, folder: Path, *, results_folder: Path | None = None) -> dict:
    """Prepare new events/snapshots first, publish news atomically, leave vectors to version-checked lazy creation."""
    from stage4_news_relevance.plausibility.events import ensure_events
    folder = Path(folder)
    if news.id.isna().any() or not news.id.is_unique or news.url.isna().any() or news.url.str.strip().eq('').any():
        raise ValueError('News require non-empty unique IDs and source URLs')
    if news.url.map(normalize_url).duplicated().any():
        raise ValueError('News have duplicate normalized URLs')
    if news.id.astype(str).tolist() != news.url.map(stable_id).astype(str).tolist():
        raise ValueError('Published IDs must use the declared URL hash scheme')
    previous_path = folder / 'news.parquet'
    previous = pd.read_parquet(previous_path) if previous_path.exists() else None
    if previous is not None:
        archive_corpus(previous, folder)
        archive_corpus(filtered_news(previous), folder)
    current = news.copy()
    fingerprint = corpus_fingerprint(current)
    current.attrs.update(corpus_fingerprint=fingerprint, id_scheme=ID_SCHEME)
    archive_corpus(current, folder)
    effective = filtered_news(current)
    snapshot = archive_corpus(effective, folder)
    prepared_events = snapshot.parent / 'events.parquet'
    ev = ensure_events(effective, prepared_events)
    migration = []
    if previous is not None:
        index = _url_index(current)
        migration = [resolve_reference(news_reference(r), index=index) for r in previous.to_dict('records')]
        write_json(snapshot.parent / f'migration_from_{corpus_fingerprint(previous)}.json', migration)
    # Readers validate embedded fingerprints; they can never accept an event/vector file from another corpus.
    _atomic(folder / 'events.parquet', lambda p: shutil.copyfile(prepared_events, p))
    _atomic(folder / 'news.jsonl', lambda p: current.to_json(p, orient='records', lines=True, force_ascii=False, date_format='iso'))
    _atomic(previous_path, lambda p: current.to_parquet(p, index=False))
    report = {'corpus_fingerprint': fingerprint, 'id_scheme': ID_SCHEME, 'news': len(current), 'events': len(ev),
              'previous_corpus': corpus_fingerprint(previous) if previous is not None else None,
              'migration': {s: sum(r['status'] == s for r in migration)
                            for s in ('current', 'remapped', 'changed', 'missing', 'ambiguous')},
              'embeddings': 'lazy: only matching corpus fingerprints and IDs may be loaded', 'saved_results': []}
    queue_manifest = folder.parent / 'llm_runs' / 'manifest.json'
    if queue_manifest.exists():
        try:
            queue_version = json.loads(queue_manifest.read_text(encoding='utf-8')).get('corpus_fingerprint')
        except (OSError, ValueError):
            queue_version = None
        report['frozen_llm_queues'] = 'current' if queue_version == fingerprint else 'unverified or stale: regenerate'
    result_dir = Path(results_folder) if results_folder is not None else folder.parent / 'insights'
    if result_dir.exists():
        for path in sorted(result_dir.glob('*.json')):
            if path.name.endswith(('.references.json', '.diagnostics.json')):
                continue
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                report['saved_results'].append({'file': path.name, 'ok': False, 'status': 'unreadable'})
                continue
            if not isinstance(payload, list):
                continue
            states = [result_references(r, current) for r in payload]
            report['saved_results'].append({'file': path.name, 'ok': all(s['ok'] for s in states)})
            write_json(path.with_suffix('.references.json'), {'corpus_fingerprint': fingerprint, 'results': states})
    write_json(folder / 'rebuild_report.json', report)
    return report


def experiment_binding(news: pd.DataFrame, ev: pd.DataFrame) -> dict:
    membership = [(str(r.event), sorted(map(str, r.news_ids))) for r in ev.itertuples()]
    return {'corpus_fingerprint': corpus_fingerprint(news), 'events_fingerprint': hashlib.sha256(
        json.dumps(sorted(membership)).encode()).hexdigest()}


def assert_experiment_corpus(path: Path) -> None:
    from common.data import DATA
    path = Path(path)
    if not path.exists():
        raise StaleNewsReferences('Experiment artifacts have no verified corpus binding')
    news_path, events_path = DATA / 'news/news.parquet', DATA / 'news/events.parquet'
    def stamp(p):
        s = p.stat() if p.exists() else None
        return (s.st_mtime_ns, s.st_size) if s else None
    _assert_experiment_corpus(str(path), str(news_path), str(events_path),
                              (stamp(path), stamp(news_path), stamp(events_path)))


@lru_cache(4)
def _assert_experiment_corpus(binding_path: str, news_path: str, events_path: str, stamps: tuple) -> None:
    from stage4_news_relevance.plausibility.events import ensure_events
    news = pd.read_parquet(news_path)
    actual = experiment_binding(news, ensure_events(filtered_news(news), Path(events_path)))
    if actual != json.loads(Path(binding_path).read_text(encoding='utf-8')):
        raise StaleNewsReferences('News/events changed: regenerate experiment embeddings, event labels and frozen samples')


if __name__ == '__main__':
    import argparse
    from common.data import DATA
    ap = argparse.ArgumentParser(description='Check saved news references without numeric-ID guessing')
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', type=Path, help='validate saved pipeline JSON against current news')
    action.add_argument('--history', type=Path, help='validate and read the archived original sources')
    action.add_argument('--bind-pairs', type=Path, help='explicitly bind a golden set while its ORIGINAL corpus is active')
    a = ap.parse_args()
    try:
        if a.history:
            print(f'{len(load_history(a.history, DATA / "news"))} results: original sources verified')
        else:
            news = pd.read_parquet(DATA / 'news' / 'news.parquet')
            if a.bind_pairs:
                print(bind_pairs(a.bind_pairs, news))
            else:
                print(f'{len(load_results(a.check, news))} results: current references verified')
    except StaleNewsReferences as exc:
        ap.exit(1, str(exc) + '\n')
