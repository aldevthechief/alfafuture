"""Blame assignment between selection and insight text, position-corrected open rate, sparse-feedback training rows."""
from monitoring.feedback import attribute, fit_position_decay, insight_signals, selection_signals, training_examples


def test_explicit_reasons_name_the_stage():
    assert attribute('not_relevant') == dict(selection=1.0, insight=0.0)
    assert attribute('already_known') == dict(selection=1.0, insight=0.0)
    assert attribute('incorrect') == dict(selection=0.0, insight=1.0)
    assert attribute('like') == dict(selection=1.0, insight=1.0)  # a positive credits both
    assert attribute('question') == dict(selection=0.0, insight=0.0)


def test_generic_dislike_follows_the_judge_then_the_behaviour():
    off_topic = dict(relevance=1, clarity=5, grounding=5)  # the event is not for them, the text is fine
    badly_written = dict(relevance=5, clarity=2, grounding=2)
    assert attribute('dislike', off_topic)['selection'] == 1.0 and attribute('dislike', badly_written)['insight'] == 1.0
    assert attribute('dislike') == dict(selection=0.5, insight=0.5)  # no information: split evenly
    assert attribute('dislike', dwell_s=2)['selection'] == 0.75  # a bounce: the card promised what the event did not give
    assert attribute('dislike', dwell_s=120)['insight'] == 0.75  # read to the end, then disliked: the text


def events():
    out = []
    for user in range(40):  # card A at position 1, card B at position 3; B is shown less prominently but opened as often
        for card, pos, opens in (('A', 1, user < 20), ('B', 3, user < 20)):
            out.append(dict(user=user, insight_id=card, kind='impression', position=pos))
            if opens:
                out.append(dict(user=user, insight_id=card, kind='open'))
    return out


def test_selection_signals_correct_for_position():
    sig = selection_signals(events(), eta=0.0)
    assert sig['A']['open_rate'] == sig['B']['open_rate'] == 0.5  # no correction: both 50%
    sig = selection_signals(events(), eta=1.0)
    assert sig['A']['open_rate'] == 0.5  # position 1: p = 1
    pos = [dict(user=u, insight_id='x', kind='impression', position=p) for u in range(10) for p in (1, 2, 3, 4)]
    pos += [dict(user=u, insight_id='x', kind='open', position=p) for u in range(10) for p in (1, 2, 3, 4) if u < 10 // p]
    assert fit_position_decay(pos) > 0.5  # opens fall with the position


def test_insight_signals_normalise_dwell_by_text_length():
    ev = [dict(user=1, insight_id='A', kind='open'), dict(user=1, insight_id='A', kind='dwell', dwell_s=50, text_chars=1250),
          dict(user=2, insight_id='A', kind='open'), dict(user=2, insight_id='A', kind='dwell', dwell_s=2, text_chars=1250),
          dict(user=1, insight_id='A', kind='source_click'), dict(user=1, insight_id='A', kind='reaction', reaction='incorrect')]
    s = insight_signals(ev)['A']
    assert s['opens'] == 2 and s['bounce_share'] == 0.5 and s['source_clicks_per_open'] == 0.5 and s['text_blame'] == 1.0
    assert abs(s['dwell_normalised'] - 0.52) < 1e-9  # median of 50 s / 50 s needed and 2 s / 50 s


def test_training_rows_use_weak_negatives_for_silence_and_strong_evidence_wins():
    ev = [dict(user=1, insight_id='A', kind='impression'), dict(user=2, insight_id='A', kind='impression'),
          dict(user=2, insight_id='A', kind='dwell', dwell_s=100, text_chars=1000),
          dict(user=3, insight_id='A', kind='impression'), dict(user=3, insight_id='A', kind='hide')]
    rows = {(u, i): (p, c) for u, i, p, c in training_examples(ev)}
    assert rows[(1, 'A')] == (0, 0.1) and rows[(2, 'A')][0] == 1 and rows[(2, 'A')][1] > 3 and rows[(3, 'A')] == (0, 5.0)
