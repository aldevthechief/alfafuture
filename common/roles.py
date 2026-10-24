"""Reader roles from roles.json: the single source for retrieval (role -> macro channels) and for the writer prompt."""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(None)
def roles() -> dict[str, dict]:
    return json.loads(Path(__file__).with_name('roles.json').read_text(encoding='utf-8'))['roles']


def role(key: str) -> dict:
    if key not in roles():
        raise ValueError(f'unknown role {key!r}; known: {", ".join(roles())}')
    return roles()[key]


def canonical_role(value: str) -> str:
    """Recognize stored display names from older model outputs; leave unknown roles unchanged."""
    if not isinstance(value, str):
        return value  # The receiving schema handles invalid types.
    normalized = value.strip().casefold()
    for key, spec in roles().items():
        title = spec['title'].casefold()
        if normalized in (key.casefold(), title, f'{title} ({key.casefold()})'):
            return key
    return value


def role_channels(key: str | None) -> list[str]:
    """Macro channel keys the role cares about; no role -> none."""
    return role(key)['channels'] if key else []


def role_context(key: str) -> dict[str, str]:
    """Prompt variables describing the reader."""
    r = role(key)
    return {'role_title': r['title'], 'role_focus': r['focus'], 'role_decisions': '; '.join(r['decisions'])}
