"""Registry for local models that are not sentence-transformers (the cross-encoder); sentence-transformers go through
common/encoders.py. Both take the same lock: transformers builds weights on the meta device process-wide while loading,
so two loads at once corrupt each other («Cannot copy out of meta tensor»)."""
from common.encoders import _LOCK

_MODELS = {}


def load_once(key, factory):
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = factory()
        return _MODELS[key]
