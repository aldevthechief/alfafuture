"""Is the golden set separable by cheap artefacts of its construction? A bag-of-words classifier trained on the OTHER companies'
items and a length baseline predict the label of one company's items. A high AUC means a retriever (or a classifier) could
win on style / vocabulary instead of on relevance to the company, so the metrics of that set would flatter it.
Reading: ~0.5 = nothing leaks; > 0.7 = the set is partly decided by wording; report the number next to every retrieval metric.
python -X utf8 -m evals.retrieval_golden.leakage"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
GOLDEN = Path(__file__).with_name('golden')


def load(folder: Path = GOLDEN) -> dict[str, list[dict]]:
    return {p.stem: json.loads(p.read_text(encoding='utf-8'))['items'] for p in sorted(folder.glob('U_*.json'))}


def _auc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None


def leakage(sets: dict[str, list[dict]], negatives: tuple[str, ...] = ('hard_neg', 'easy_neg')) -> dict:
    """Leave-one-company-out: train on the others, score the held-out company; AUC pooled over companies."""
    text = lambda i: i['title'] + '. ' + i['text']
    out = {}
    for name, kinds in (('all_negatives', negatives), ('hard_negatives', ('hard_neg',))):
        y, bow, length = [], [], []
        for held in sets:
            train = [i for c, items in sets.items() if c != held for i in items if i['label'] == 1 or i['kind'] in kinds]
            test = [i for i in sets[held] if i['label'] == 1 or i['kind'] in kinds]
            if len({i['label'] for i in train}) < 2 or not test:
                continue
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            clf = LogisticRegression(max_iter=1000, C=2.0).fit(vec.fit_transform(text(i) for i in train), [i['label'] for i in train])
            y += [i['label'] for i in test]
            bow += list(clf.predict_proba(vec.transform(text(i) for i in test))[:, 1])
            length += [len(i['text']) for i in test]
        out[name] = dict(n=len(y), bag_of_words_auc=_auc(y, bow), length_auc=_auc(y, length))
    flat = [i for items in sets.values() for i in items if i['kind'] == 'pos']
    out['positives_naming_industry'] = float(np.mean([i['names_industry'] for i in flat])) if flat else None
    return out


if __name__ == '__main__':
    print(json.dumps(leakage(load()), ensure_ascii=False, indent=1))
