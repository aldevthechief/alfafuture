"""Twin tables: the same macro table repeats in several reports with other dates and values («Основные макроэкономические
показатели» in the July and September macro statistics and in the 11.09 rate report). A chunk's twin is the most similar chunk of
every other report when both are tables with the same title (month names, digits and punctuation dropped) and the e5 cosine of
the indexed chunks is at least MIN_COS. The title guard keeps similar-looking but different tables apart: cosine alone is high
for any two macro tables (0.91–0.98, and 0.95+ for 6 pairs of different tables), the real twins span 0.89–0.99. Text chunks have no
twins: the largest 4-gram overlap between texts of different reports is 0.23. Checked on the 9 reports (search_eval.md)."""
import re

import numpy as np

MIN_COS = 0.89
MONTHS = r'январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]\w*|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*|[1-4]п\d*|\d+м\d*|\d+к\d*'


def table_title(text: str) -> str:
    line = re.sub(r'\[таблица сверена[^\]]*\]', '', text.split('\n')[0].lower())
    line = re.sub(r'^\s*табл(ица|\.)?\s*\d*\s*:?', '', line)
    line = re.sub(rf'\b({MONTHS})\b|\d+|\bг\b', ' ', line)
    return re.sub(r'[^\wа-яё]+', ' ', line).strip()


def find_twins(chunks: list[dict], vectors: np.ndarray, min_cos: float = MIN_COS) -> dict[int, list[int]]:
    """chunk index -> indexes of its twins, at most one per other report, best first."""
    titles = {i: table_title(c['text']) for i, c in enumerate(chunks) if c['kind'] == 'table'}
    by_title = {}
    for i, t in titles.items():
        if len(t) >= 8:
            by_title.setdefault(t, []).append(i)
    out = {}
    for group in by_title.values():
        for i in group:
            best = {}
            for j in group:
                if chunks[j]['report'] != chunks[i]['report']:
                    s = float(vectors[i] @ vectors[j])
                    if s >= min_cos and s > best.get(chunks[j]['report'], (0, 0))[0]:
                        best[chunks[j]['report']] = (s, j)
            if best:
                out[i] = [j for _, j in sorted(best.values(), reverse=True)]
    return out
