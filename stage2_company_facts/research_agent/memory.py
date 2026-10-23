"""Agent memory: the durable result (facts + hypotheses) and an archive of raw tool observations.

Memory lives in the LangGraph state (so the checkpointer saves it) and is never summarized away: after context
compaction it is re-injected verbatim, and any dropped observation can be re-read by id (`recall_observation`).
The pydantic models at the bottom are the memory tools the model sees; the graph executes them against `Memory`.
"""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STATUSES = ('open', 'supported', 'plausible', 'weak', 'refuted')


class Memory:
    def __init__(self, facts: list[dict] | None = None, hypotheses: list[dict] | None = None,
                 observations: dict[str, dict] | None = None):
        self.facts = [dict(f) for f in facts or []]
        self.hypotheses = [dict(h, notes=list(h['notes'])) for h in hypotheses or []]
        self.observations = dict(observations or {})

    @classmethod
    def from_state(cls, state: dict) -> 'Memory':
        return cls(state.get('facts'), state.get('hypotheses'), state.get('observations'))

    def to_state(self) -> dict:
        return {'facts': self.facts, 'hypotheses': self.hypotheses, 'observations': self.observations}

    def observe(self, tool: str, args: dict, text: str) -> str:
        oid = f'obs{len(self.observations) + 1}'
        self.observations[oid] = {'tool': tool, 'args': args, 'text': text}
        return oid

    def note_fact(self, text: str, evidence: str = '') -> str:
        if any(f['text'] == text for f in self.facts):
            return 'duplicate fact ignored'
        self.facts.append({'id': f'F{len(self.facts) + 1}', 'text': text, 'evidence': evidence})
        return f'stored {self.facts[-1]["id"]}'

    def propose_hypothesis(self, text: str, evidence_for: str = '', evidence_against: str = '', test: str = '') -> str:
        h = {'id': f'H{len(self.hypotheses) + 1}', 'text': text, 'for': evidence_for, 'against': evidence_against,
             'test': test, 'status': 'open', 'notes': []}
        self.hypotheses.append(h)
        return f'stored {h["id"]} (status open; verify it with verify_hypothesis)'

    def hypothesis(self, hid: str) -> dict | None:
        return next((h for h in self.hypotheses if h['id'] == hid), None)

    def set_status(self, hid: str, status: str, note: str = '') -> str:
        h = self.hypothesis(hid)
        if not h or status not in STATUSES:
            return f'unknown hypothesis or status; statuses: {STATUSES}'
        h['status'] = status
        if note:
            h['notes'].append(note)
        return f'{hid} -> {status}'

    def open_hypotheses(self) -> list[str]:
        return [h['id'] for h in self.hypotheses if h['status'] == 'open']

    def recall(self, oid: str) -> str:
        o = self.observations.get(oid)
        return json.dumps(o, ensure_ascii=False) if o else 'no such observation'

    def render(self) -> str:
        facts = '\n'.join(f'{f["id"]}: {f["text"]} [{f["evidence"]}]' for f in self.facts) or '(none)'
        hyps = '\n'.join(f'{h["id"]} [{h["status"]}]: {h["text"]} | for: {h["for"]} | against: {h["against"]} | '
                         f'test: {h["test"]} | notes: {"; ".join(h["notes"])}' for h in self.hypotheses) or '(none)'
        return f'MEMORY — new facts:\n{facts}\n\nhypotheses:\n{hyps}'


class ProposeHypothesis(BaseModel):
    """Store a hypothesis about the business with evidence for / against and a refutation test."""
    model_config = ConfigDict(title='propose_hypothesis')
    text: str
    evidence_for: str = ''
    evidence_against: str = ''
    test: str = Field('', description='query whose result could refute the hypothesis')


class VerifyHypothesis(BaseModel):
    """Run the refutation-first verification sub-loop for one open hypothesis."""
    model_config = ConfigDict(title='verify_hypothesis')
    hid: str


class RecallObservation(BaseModel):
    """Re-read a raw tool observation that was truncated or dropped by context compaction."""
    model_config = ConfigDict(title='recall_observation')
    oid: str


class NewFact(BaseModel):
    text: str = Field(description='fact that follows directly from tool outputs and is NOT in the known-facts list')
    evidence: str = Field(description='obs-ids the fact follows from')


class Finish(BaseModel):
    """Finish the research: the new facts (strictly absent from the known-facts list of the task) and a short summary
    of the business model with the ids of the hypotheses behind it. Rejected while hypotheses are open."""
    model_config = ConfigDict(title='finish')
    new_facts: list[NewFact] = Field(default_factory=list, description='only facts absent from the known-facts list')
    summary: str = Field(description='business model in 2-4 sentences, citing hypothesis ids (H1, H2...)')


class Verdict(BaseModel):
    """Final verdict for the hypothesis under verification."""
    model_config = ConfigDict(title='report_verdict')
    status: Literal['supported', 'plausible', 'weak', 'refuted']
    reason: str = Field(description='short justification with obs-ids')


MEMORY_TOOLS = [ProposeHypothesis, VerifyHypothesis, RecallObservation, Finish]
