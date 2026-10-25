"""The labelling task of the teammate (insight_judge_calibration/*.md): exposure + 14 pairs per company, and their answers
(marking_sheet.csv). Answers typed on a Russian keyboard use Cyrillic А/В: they are read as A/B."""
import csv
import re
from dataclasses import dataclass
from pathlib import Path

AXES = ('relevance', 'significance', 'clarity', 'grounding')
PAIR = re.compile(r'### Пара (\S+)\n\*\*A\.\*\*\n(.*?)\n\n\*\*B\.\*\*\n(.*?)\n\n---', re.S)
CYR = {'А': 'A', 'В': 'B', 'a': 'A', 'b': 'B', '＝': '=', '≈': '='}


@dataclass(frozen=True)
class Pair:
    id: str
    company: str       # U_14366
    role: str          # cfo
    exposure: str
    a: str
    b: str


def load_pairs(folder: Path) -> list[Pair]:
    pairs = []
    for path in sorted(Path(folder).glob('U_*.md')):
        text = path.read_text(encoding='utf-8')
        company, role = '_'.join(path.stem.split('_')[:2]), path.stem.split('_')[-1]  # U_14366_cfo.md
        head = text.split('## Пары для сравнения')[0]
        exposure = head.split('\n', 2)[2].strip().rstrip('-').strip()  # without the file title
        for pid, a, b in PAIR.findall(text):
            pairs.append(Pair(pid, company, role, exposure, a.strip(), b.strip()))
    return pairs


def norm(value: str) -> str | None:
    v = CYR.get((value or '').strip(), (value or '').strip().upper())
    return v if v in ('A', 'B', '=') else None


def load_human(sheet: Path) -> dict[str, dict]:
    """pair id -> {axis: A/B/=, winner, confidence, factual_error, comment}; rows not filled yet are skipped."""
    out = {}
    with open(sheet, encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if not row or row[0] == 'pair_id' or len(row) < 6:
                continue
            row = row + [''] * (9 - len(row))
            labels = [norm(x) for x in row[1:6]]
            if None in labels:
                continue  # an empty or invalid cell: the pair is not labelled
            out[row[0].strip()] = dict(zip((*AXES, 'winner'), labels),
                                       confidence=int(row[6]) if row[6].strip().isdigit() else None,
                                       factual_error=row[7].strip() == '1', comment=row[8])
    return out
