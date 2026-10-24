"""One process-wide cache and initialization lock for local sentence encoders.

Transformers temporarily patches torch's parameter registration while loading weights onto meta.
Overlapping constructors (even for different models) can restore another thread's temporary patch.
Only construction is serialized; already loaded encoders can run inference concurrently.
"""
import json
import threading

_LOCK = threading.RLock()
_MODELS = {}


def load_encoder(name: str, *, device: str | None = None, model_kwargs=None,
                 tokenizer_kwargs=None, max_seq_length: int = 512):
    import torch

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model_kwargs = dict(model_kwargs or {})
    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    if device == 'cuda':
        model_kwargs.setdefault('torch_dtype', torch.float16)
    key = (name, device, max_seq_length,
           json.dumps(model_kwargs, sort_keys=True, default=str),
           json.dumps(tokenizer_kwargs, sort_keys=True, default=str))
    with _LOCK:
        if key not in _MODELS:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(name, device=device, model_kwargs=model_kwargs,
                                        tokenizer_kwargs=tokenizer_kwargs)
            # Never cache an unmaterialized model or "repair" it with random/uninitialized weights.
            if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
                raise RuntimeError(f'Encoder {name} has unloaded meta tensors after loading checkpoint weights')
            model.max_seq_length = max_seq_length
            _MODELS[key] = model
        return _MODELS[key]


def clear_encoder_cache():
    with _LOCK:
        _MODELS.clear()
