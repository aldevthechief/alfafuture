"""Confidential-data tripwires for reader text, two separate signals (the 25.10 review of the transport insights):
- `peer_hits`: the bank's view of OTHER companies shown to this client — peer ranks, counts and medians of its OKVED group
  («1-е место среди 1082 компаний», «медиана по разделу 49: 7%»). Any hit is a leak.
- `client_hits`: the client's own revenue / payment shares / counterparties next to a figure. Not a leak to a third party, but
  «73% исходящих платежей …» in a reader card shows the bank watching the account; the rate is tracked, the wording is a
  product decision.
Deterministic tripwires, not a judge: they find the wordings seen so far, so a zero rate does not prove there is no leak —
sample what they miss (human spot check, docs/QUALITY_AND_MONITORING.md)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage6_insight_generation.insight_compact.compact import COMPANY_DATA, RANK

MEDIAN = re.compile(r'медиан\w*', re.I)


def _sentences(text: str) -> list[str]:
    return re.split(r'(?<=[.;!?])\s+', text or '')


def peer_hits(text: str) -> list[str]:
    return [s for s in _sentences(text) if RANK.search(s) or MEDIAN.search(s)]


def client_hits(text: str) -> list[str]:
    return [s for s in _sentences(text) if COMPANY_DATA.search(s) and not (RANK.search(s) or MEDIAN.search(s))]
