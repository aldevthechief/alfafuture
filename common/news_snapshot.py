"""Context-local immutable corpus selection for the entire analysis, including nested tools."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

ACTIVE = ContextVar('news_snapshot', default=None)

@contextmanager
def news_snapshot(news=None):
    current = ACTIVE.get()
    if current is not None:
        yield current
        return
    state = {'news': news, 'index': None}
    token = ACTIVE.set(state)
    try:
        yield state
    finally:
        ACTIVE.reset(token)

def pinned_run(function):
    @wraps(function)
    def run(*args, **kwargs):
        with news_snapshot():
            return function(*args, **kwargs)
    return run
