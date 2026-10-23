"""Stage-2 profile for the pipeline: from data/companies/profiles.jsonl if the company is there, else built on the fly
(any company of the tables, ~2 s). `brief()` = the lines a prompt gets about why news can matter to the company."""
import json
from functools import lru_cache
from pathlib import Path
from stat import S_ISREG

from common.data import DATA

PROFILES = DATA / 'companies' / 'profiles.jsonl'


def _file(path: Path | None = None) -> dict[str, dict]:
    """Check the file version on each profile lookup; unchanged files reuse the parsed profiles."""
    path = Path(PROFILES if path is None else path).resolve()
    try:
        info = path.stat()
    except FileNotFoundError:
        return {}
    if not S_ISREG(info.st_mode):
        return {}
    version = (info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino)
    return _read_file(path, version)


@lru_cache(maxsize=1)
def _read_file(path: Path, version: tuple[int, int, int, int]) -> dict[str, dict]:
    """The version is part of the cache key, including file identity for atomic replacements."""
    with open(path, encoding='utf-8') as f:
        return {p['id']: p for p in map(json.loads, filter(str.strip, f))}


def load_profile(company_id: str) -> dict:
    p = _file().get(company_id)
    if p is None:
        from stage2_company_facts.profile.profile import profiles
        p = profiles([company_id]).get(company_id)
        if p is None:
            raise KeyError(f'{company_id} is not in the case tables')
    return p


def brief(p: dict, n_levers: int = 6, n_factors: int = 6) -> list[str]:
    """Scale, strongest levers and external factors with their mechanism — appended to the key facts of prompts."""
    s = p['scale']
    lines = [f'[scale] Масштаб: {s["class_ru"]} (основание: {s["class_basis"]})'
             + ('; крупный бизнес по критерию кейса' if s['case_large'] else '')
             + ('; группа — крупный бизнес по критерию кейса' if s['group_case_large'] and not s['case_large'] else '')]
    top = sorted(p['sensitivity'].values(), key=lambda x: -x['weight'])[:n_levers]
    lines.append('[sensitivity] Чувствительность к рычагам: ' + ', '.join(f'{x["lever"]} {x["name"]} {x["weight"]:.1f}' for x in top))
    for f in p['external_factors'][:n_factors]:
        lines.append(f'[factor {f["weight"]:.1f}] {f["label"]}: {f["mechanism"]}')
    sup = p.get('links', {}).get('suppliers') or {}
    if sup.get('hhi') and sup['hhi'] >= 0.25:
        lines.append(f'[links] Высокая зависимость от поставщиков: крупнейший {sup["top1_share"]:.0%}, топ-5 {sup["top5_share"]:.0%} '
                     'исходящих платежей')
    return lines
