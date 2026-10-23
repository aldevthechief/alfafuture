"""Agentic factory of candidate generators (experiments/generator_factory, prompt v2 + few-shots = run «B»): GLM reads the
deterministic company brief, writes a business-model card, walks 24 channels of influence, checks its own queries on the
news pool with `probe` and finishes by picking probes by number with a coverage report per channel, so nothing unprobed
gets in. The result (card, probes, picks) is cached per company: queries describe the company, not the news, so a new
news pool is ranked with the same generators without new LLM calls."""
import hashlib, json, sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from common.data import DATA
from common.llm import ChatModels, PromptLibrary, ToolLoop
from stage3_news_search.retrieval_v2.brief import deterministic

PROMPTS = PromptLibrary(Path(__file__).with_name('prompts'))
SPECS = DATA / 'retrieval_v2' / 'factory'
VERSION = 'v2-B'  # prompt v2 + few-shots; bump to rebuild the cached specs
QUOTA = 13  # events per generator: ~16 generators -> ~200 events


class Pick(BaseModel):
    probe_id: int = Field(description='номер пробы из ответа probe')
    channel: str = Field(description='id канала, например A2')
    reason: str = Field(description='короткое имя повода латиницей')
    fact: str = Field(description='факт профиля, через который канал касается компании')


class Coverage(BaseModel):
    channel: str
    probe_ids: list[int] = []
    why_none: str = Field('', description='почему генератора нет: нет факта / в пуле новостей нет')


class Finish(BaseModel):
    """Selected probes and coverage of every channel."""
    model_config = ConfigDict(title='finish')
    selected: list[Pick]
    coverage: list[Coverage]


class Factory:
    def __init__(self, search, model, max_steps: int = 40):
        self.search, self.model, self.max_steps = search, model, max_steps
        self.probes: list[dict] = []
        self.card: dict = {}

    def probe(self, instruction: str, queries: list[str]) -> str:
        """Check a generator: its number, top-10 events (date, title) and how many of its top-13 earlier probes did not find."""
        queries = queries[:3]
        top = self.search.rank_events(instruction, queries, QUOTA)
        seen = set().union(*(p['top'] for p in self.probes))
        pid = len(self.probes)
        self.probes.append({'probe_id': pid, 'instruction': instruction, 'queries': queries, 'top': top})
        ev = self.search.events
        lines = [f'{str(ev.first_date[e])[:10]} | {ev.title[e][:110]}' for e in top[:10]]
        return f'probe_id={pid}\n' + '\n'.join(lines) + f'\nновых событий в топ-{QUOTA} относительно прежних проб: {len(set(top) - seen)}'

    def note_business_model(self, products: str, buyers: str, costs_and_suppliers: str, regulation: str,
                            ownership_and_group: str, hypotheses: str) -> str:
        """Write down the company's business model: facts from the profile + industry knowledge (the latter listed in hypotheses)."""
        self.card = dict(products=products, buyers=buyers, costs_and_suppliers=costs_and_suppliers, regulation=regulation,
                         ownership_and_group=ownership_and_group, hypotheses=hypotheses)
        return 'записано'

    def run(self, facts: list[dict]) -> dict | None:
        """None if the agent did not call finish within max_steps."""
        system, task = (m.content for m in PROMPTS.chat('factory').format_messages(
            brief=deterministic(facts), fewshots=PROMPTS.text('factory_fewshots')))
        tools = [StructuredTool.from_function(self.probe), StructuredTool.from_function(self.note_business_model)]
        res = ToolLoop(self.model, tools, Finish, system, self.max_steps,
                       nudge='Проверь генераторы через probe или вызови finish.').run(task)
        if res is None:
            return None
        bad = [p.probe_id for p in res.selected if not 0 <= p.probe_id < len(self.probes)]
        return {'version': VERSION, 'card': self.card, 'probes': self.probes, 'bad_probe_ids': bad,
                'selected': [p.model_dump() for p in res.selected if p.probe_id not in bad],
                'coverage': [c.model_dump() for c in res.coverage]}


MAX_STEPS = 40
TEMPERATURE = 0.2


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def model_id(model) -> str:
    """Model name and sampling; the default agent is named from settings, so a cache hit needs no API key."""
    if model is None:
        return f'{ChatModels().settings.agent_model}|t={TEMPERATURE}'
    return f"{getattr(model, 'model_name', type(model).__name__)}|t={getattr(model, 'temperature', None)}"


def spec_input(facts: list[dict], search, model) -> dict[str, str]:
    """Everything the agent's result depends on; a cached spec is reused only if every part is equal: the company facts
    and the brief built from them, the prompt texts, the version, the model and its sampling, the step cap and probe
    quota, and the news pool with the search configuration (probes see the pool: a new pool can open channels the agent
    found empty)."""
    return {'version': VERSION, 'facts': _sha(json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)),
            'brief': _sha(deterministic(facts)),
            'prompts': _sha(PROMPTS.text('factory') + '\x00' + PROMPTS.text('factory_fewshots')),
            'model': model_id(model),
            'max_steps': str(MAX_STEPS), 'quota': str(QUOTA), 'pool': search.fingerprint}


def factory_spec(company_id: str, facts: list[dict], search, model=None, refresh: bool = False,
                 frozen: bool = False) -> dict | None:
    """Cached spec (data/retrieval_v2/factory/<company>.json, gitignored: it quotes case facts); runs the agent (GLM,
    ~4 ₽) when any part of spec_input differs — the changed parts are reported on stderr.
    frozen (evaluation on old labels only): return the stored spec whatever its input, never run the agent."""
    path = SPECS / f'{company_id}.json'
    key = spec_input(facts, search, model)
    if path.exists() and not refresh:
        spec = json.loads(path.read_text(encoding='utf-8'))
        if spec.get('input') == key:
            return spec
        old = spec.get('input') or {}
        changed = [k for k in key if old.get(k) != key[k]]
        print(f'factory cache miss for {company_id}: changed {", ".join(changed)}'
              + ('; frozen: the stored spec is used' if frozen else '; running the agent'), file=sys.stderr)
        if frozen:
            return spec
    if frozen:
        return None
    spec = Factory(search, model or ChatModels().agent(temperature=TEMPERATURE), MAX_STEPS).run(facts)
    if spec is not None:  # a failed run is not cached: the next call retries
        spec['input'] = key
        SPECS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding='utf-8')
    return spec


def generators(spec: dict) -> list[dict]:
    """Selected probes, one generator per probe id (several channels may pick the same probe)."""
    out = {}
    for p in spec['selected']:
        pr = spec['probes'][p['probe_id']]
        out.setdefault(p['probe_id'], {'reason': p['reason'], 'channel': p['channel'], 'fact': p['fact'],
                                       'instruction': pr['instruction'], 'queries': pr['queries']})
    return list(out.values())
