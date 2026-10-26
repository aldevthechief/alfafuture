"""Who is to blame for a user's reaction: the SELECTION of the event (retrieval + ranking + the judge's «is it an insight») or the
INSIGHT text (writer, grounding)? Design and the recsys background: docs/QUALITY_AND_MONITORING.md.

Events (one list of dicts per user action; the UI does not emit impressions / dwell yet — see the doc):
  {user, company, role, insight_id, kind, ts, position?, dwell_s?, text_chars?, reaction?}
  kind: impression | open | dwell | reaction | hide | question | source_click
  reaction (feedback_api kinds): like useful dislike not_relevant already_known incorrect

1. Explicit reason  -> blame by what the reaction says (RULES).
2. Generic dislike  -> a prior from the offline judge (relevance low -> selection; clarity / grounding low -> insight), then the
   behaviour: a bounce right after opening points at selection / headline, a long read before the dislike at the text.
3. Positive         -> both stages get credit (the event mattered AND the text worked).
Selection is also measured without any reaction: open rate per impression, corrected for the position (SNIPS), unobserved
impressions as weak negatives (Hu–Koren–Volinsky confidence), normalised dwell for the insight."""
from collections import defaultdict

import numpy as np

# reaction -> (selection, insight, why). Weights sum to 1 for a negative reaction; a positive one credits both.
RULES = {
    'not_relevant': (1.0, 0.0, 'the event is not for this company / role'),
    'already_known': (1.0, 0.0, 'novelty: the ranker showed what the user knew'),
    'incorrect': (0.0, 1.0, 'a wrong fact or number in the text'),
    'like': (1.0, 1.0, 'both worked'),
    'useful': (1.0, 1.0, 'both worked'),
}
BOUNCE_S = 5          # left the opened insight before this: the card promised something the event did not give
READ_CHARS_PER_S = 25  # ~1500 characters per minute: a rough reading speed to normalise dwell by the text length
UNSEEN_WEIGHT = 0.1   # an impression with no action: weak evidence of disinterest, not a label (lack of exposure != dislike)


def attribute(reaction: str, judge: dict | None = None, dwell_s: float | None = None) -> dict:
    """{'selection': w, 'insight': w} for one reaction. `judge` = the offline judge's 1–5 axis scores of this insight."""
    if reaction in RULES:
        s, i, _ = RULES[reaction]
        return dict(selection=s, insight=i)
    if reaction != 'dislike':
        return dict(selection=0.0, insight=0.0)
    sel, ins = 0.5, 0.5
    if judge:  # the calibrated judge says where the weakness is
        sel_gap = 5 - judge['relevance']
        ins_gap = ((5 - judge['clarity']) + (5 - judge['grounding'])) / 2
        sel, ins = (0.5, 0.5) if sel_gap + ins_gap == 0 else (sel_gap / (sel_gap + ins_gap), ins_gap / (sel_gap + ins_gap))
    if dwell_s is not None:  # behaviour moves the prior, it does not replace it
        shift = 0.25 if dwell_s < BOUNCE_S else -0.25 if dwell_s > 60 else 0.0
        sel = min(1.0, max(0.0, sel + shift))
        ins = 1.0 - sel
    return dict(selection=round(sel, 3), insight=round(ins, 3))


def fit_position_decay(events: list[dict]) -> float:
    """eta of p(position) = position^-eta from the open rate per position (log-log slope). Confounded: the ranker puts the better
    cards first, so this OVERestimates the decay — calibrate with randomised swaps of neighbours (a few % of sessions)."""
    shown, opened = defaultdict(int), defaultdict(int)
    for e in events:
        if e.get('position') and e['kind'] in ('impression', 'open'):
            (shown if e['kind'] == 'impression' else opened)[e['position']] += 1
    pos = [p for p in sorted(shown) if shown[p] >= 5 and opened[p] > 0]
    if len(pos) < 3:
        return 0.0
    slope = np.polyfit(np.log(pos), np.log([opened[p] / shown[p] for p in pos]), 1)[0]
    return float(max(0.0, -slope))


def selection_signals(events: list[dict], eta: float = 0.0) -> dict[str, dict]:
    """Per insight: impressions, opens, the position-corrected open rate (SNIPS: sum(open / p) / sum(1 / p) over impressions, as if
    every card were shown at position 1) and the explicit selection blame / credit. p(position) = position^-eta."""
    rows = defaultdict(lambda: dict(impressions=0, opens=0, w_open=0.0, w_all=0.0, credit=0.0, blame=0.0))
    opened = {(e['user'], e['insight_id']) for e in events if e['kind'] == 'open'}
    for e in events:
        r = rows[e['insight_id']]
        if e['kind'] == 'impression':
            inv = (e.get('position') or 1) ** eta  # 1 / p(position)
            hit = (e['user'], e['insight_id']) in opened
            r['impressions'] += 1
            r['opens'] += hit
            r['w_open'] += inv * hit
            r['w_all'] += inv
        elif e['kind'] in ('reaction', 'hide'):
            w = attribute('not_relevant' if e['kind'] == 'hide' else e['reaction'], e.get('judge'), e.get('dwell_s'))['selection']
            positive = e.get('reaction') in ('like', 'useful')
            r['credit' if positive else 'blame'] += w
    return {k: dict(v, open_rate=v['w_open'] / v['w_all'] if v['w_all'] else None) for k, v in rows.items()}


def insight_signals(events: list[dict]) -> dict[str, dict]:
    """Per insight, among the users who opened it: reading time normalised by the text length (1 = the time to read it once),
    bounce share, source clicks and questions per open, and the explicit blame of the text («incorrect» + generic dislikes)."""
    rows = defaultdict(lambda: dict(opens=0, dwell=[], bounces=0, source_clicks=0, questions=0, text_blame=0.0, text_credit=0.0))
    for e in events:
        r = rows[e['insight_id']]
        if e['kind'] == 'open':
            r['opens'] += 1
        elif e['kind'] == 'dwell':
            need = (e.get('text_chars') or 0) / READ_CHARS_PER_S
            r['dwell'].append(e['dwell_s'] / need if need else None)
            r['bounces'] += e['dwell_s'] < BOUNCE_S
        elif e['kind'] == 'source_click':
            r['source_clicks'] += 1
        elif e['kind'] == 'question':
            r['questions'] += 1
        elif e['kind'] == 'reaction':
            w = attribute(e['reaction'], e.get('judge'), e.get('dwell_s'))['insight']
            r['text_credit' if e['reaction'] in ('like', 'useful') else 'text_blame'] += w
    out = {}
    for k, v in rows.items():
        norm = [d for d in v['dwell'] if d is not None]
        out[k] = dict(opens=v['opens'], dwell_normalised=float(np.median(norm)) if norm else None,
                      bounce_share=v['bounces'] / len(v['dwell']) if v['dwell'] else None,
                      source_clicks_per_open=v['source_clicks'] / v['opens'] if v['opens'] else None,
                      questions_per_open=v['questions'] / v['opens'] if v['opens'] else None,
                      text_blame=v['text_blame'], text_credit=v['text_credit'])
    return out


def training_examples(events: list[dict], alpha: float = 2.0) -> list[tuple[str, str, int, float]]:
    """Sparse feedback as (user, insight_id, preference 0/1, confidence) for a ranker (Hu–Koren–Volinsky): an unobserved impression
    is a weak 0, an open a 1 whose confidence grows with the normalised read, a bounce or an explicit «not for me» a confident 0.
    One row per (user, insight): the strongest evidence wins."""
    best: dict[tuple[str, str], tuple[int, float]] = {}
    put = lambda e, pref, conf: best.update({(e['user'], e['insight_id']): max(best.get((e['user'], e['insight_id']), (0, 0.0)),
                                                                                  (pref, conf), key=lambda x: x[1])})
    for e in events:
        k = e['kind']
        if k == 'impression':
            put(e, 0, UNSEEN_WEIGHT)
        elif k == 'dwell':
            need = (e.get('text_chars') or 0) / READ_CHARS_PER_S
            if e['dwell_s'] < BOUNCE_S:
                put(e, 0, 1.0)
            else:
                put(e, 1, 1.0 + alpha * min(e['dwell_s'] / need if need else 1.0, 3.0))
        elif k == 'hide' or e.get('reaction') in ('not_relevant', 'already_known'):
            put(e, 0, 1.0 + alpha * 2)
        elif e.get('reaction') in ('like', 'useful'):
            put(e, 1, 1.0 + alpha * 2)
    return [(u, i, p, c) for (u, i), (p, c) in best.items()]
