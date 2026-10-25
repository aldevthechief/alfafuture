"""Company-wide dialogue: separate context and history, the same validated discussion engine."""
import json
import re
from pathlib import Path

from common.evidence import EvidenceSnapshot, FactEvidence
from feedback_api.analyzer import _replace_refs
from feedback_api.research import ResearchTools
from feedback_api.schemas import CompanyChatContext


def load_company_context(store, user_id, company_id, role):
    from stage2_company_facts.profile.integration import load_company
    company = load_company(company_id)
    personalization = store.personalization(user_id, company_id, role)
    extra = [*personalization.answer_facts(), *[
        dict(cat='user_context', text=f'Со слов пользователя (не проверено банком): [{note.kind}] {note.text}')
        for note in personalization.notes]]
    facts = [*company.facts[:512-len(extra)], *extra]
    evidence = EvidenceSnapshot(company_id=company_id, role=role, facts=[
        FactEvidence(id=f'fact:{n}', text=f'[{fact["cat"]}] {fact["text"]}') for n, fact in enumerate(facts[:512], 1)])
    user_context = '\n'.join([*(f'[{note.kind}] {note.text}' for note in personalization.notes),
                              *(answer.fact() for answer in personalization.answers)])
    return CompanyChatContext(card=company.card, evidence=evidence, user_context=user_context)


def analyze_company(processor, conversation, user_id, message, history, on_response=None, research_factory=None):
    from common.llm import PromptLibrary
    ctx = message.context
    history = [row for row in history if row.id != message.id][-8:]
    aliases = {f'user:{row.id}':f'user:history:{i}' for i, row in enumerate(history)}
    aliases[f'user:{message.id}'] = 'user:current'
    restore = {value:key for key, value in aliases.items()}

    def prompt_message(row):
        return _replace_refs(dict(question=row.comment, selected_answer_excerpt=row.selected_text,
                                  date=row.created_at[:10], answer=row.analysis.response if row.analysis else None), aliases)

    messages = PromptLibrary(Path(__file__).with_name('prompts')).chat('company_chat').format_messages(
        company_id=conversation.company_id, role=conversation.role,
        card=json.dumps(ctx.card, ensure_ascii=False), evidence=ctx.evidence.render(),
        user_context=ctx.user_context or '(нет уточнений)',
        history=json.dumps([{**prompt_message(row), 'id':aliases[f'user:{row.id}']} for row in history], ensure_ascii=False),
        question=json.dumps(prompt_message(message), ensure_ascii=False))
    # No event or insight is registered or loaded. Tools can only read this company's data.
    factory = (lambda: research_factory(conversation)) if research_factory else (
        lambda: ResearchTools(company_id=conversation.company_id))
    context = dict(mode='company_chat', user_id=user_id, company_id=conversation.company_id, role=conversation.role,
                   card=ctx.card, evidence=ctx.evidence.model_dump(exclude={'captured_at'}),
                   user_context=ctx.user_context, question=prompt_message(message),
                   history=[prompt_message(row) for row in history])
    return processor.complete(messages, context, factory, set(restore) | set(ctx.evidence.sources()), restore,
                              on_response=on_response)


def public_message(message):
    """Keep the full snapshot in SQLite; send only the facts cited by this answer to the browser."""
    data = message.model_dump(mode='json', exclude={'context'})
    ids = set(message.analysis.source_ids if message.analysis else [])
    if message.analysis:
        ids.update(re.findall(r'fact:\d+', message.analysis.response))
    evidence = message.context.evidence
    data['evidence'] = dict(company_id=evidence.company_id, role=evidence.role, captured_at=evidence.captured_at,
                            facts=[fact.model_dump() for fact in evidence.facts if fact.id in ids])
    return data
