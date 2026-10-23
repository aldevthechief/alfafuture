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


def role_channels(key: str | None) -> list[str]:
    """Macro channel keys the role cares about; no role -> none."""
    return role(key)['channels'] if key else []


def role_context(key: str) -> dict[str, str]:
    """Prompt variables describing the reader."""
    r = role(key)
    return {'role_title': r['title'], 'role_focus': r['focus'], 'role_decisions': '; '.join(r['decisions'])}
