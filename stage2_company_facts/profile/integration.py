"""Profile adapter shared by the pipeline engines; retrieval keeps its existing facts contract."""
from dataclasses import dataclass

from stage2_company_facts.profile.load import brief, load_profile


@dataclass(frozen=True)
class CompanyInputs:
    card: dict
    facts: list[dict]


def load_company(company_id: str) -> CompanyInputs:
    """Load once so a run's card, facts and explanatory context use the same profile version.

    Context uses a separate category: existing retrieval generators still see the original fact categories.
    Rule-derived factors and sensitivities are explicitly labelled as hypotheses, including in their text.
    """
    return company_inputs(load_profile(company_id))


def company_inputs(profile: dict) -> CompanyInputs:
    """Adapt an already-loaded profile without rebuilding it for a second API projection."""
    facts = [dict(f) for f in profile['facts']]
    for i, line in enumerate(brief(profile), 1):
        hypothesis = line.startswith(('[sensitivity]', '[factor '))
        facts.append({'id': f'fact:profile_context_{i}', 'cat': 'profile_context',
                      'text': ('Гипотеза по правилам профиля: ' if hypothesis else '') + line,
                      'depth': 1, 'entities': [], 'source': 'stage2_company_facts.profile',
                      'kind': 'hypothesis' if hypothesis else 'fact'})
    return CompanyInputs(card=profile['card'], facts=facts)


def facts_from_profiles(company_ids) -> dict[str, list[dict]]:
    return {company_id: load_company(company_id).facts for company_id in company_ids}
