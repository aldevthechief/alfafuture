"""Kaggle kernel body: score every (news, checklist item, query variant) pair with each model.
Cross-encoders read the pair in one context (query = item, document = news) -> sigmoid(logit). Bi-encoders put the
item into the instruct query and the news into the document -> cosine. CONFIG (news texts, items, variants, models)
is injected by push.py. Writes /kaggle/working/<slug>.npz: keys, items, variants, score [news, item, variant], raw.
Each model runs in its own subprocess pinned to its GPU lane; a failing model is logged and skipped."""
import gc, json, os, subprocess, sys, time, traceback

CONFIG = json.loads(r'''__CONFIG__''')
OUT = os.environ.get('OUT', '/kaggle/working')  # local smoke test overrides


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_texts():
    """news text = title + first text_chars of the body, as channels.news_text builds it locally."""
    import glob
    import pandas as pd
    df = pd.read_parquet(os.environ.get('NEWS') or glob.glob('/kaggle/input/**/news.parquet', recursive=True)[0])
    df = df.set_index(df['id'].astype(str))
    for n in CONFIG['news']:
        r = df.loc[n['id']]
        n['text'] = f"{r.title}. {(r.text or '')[:CONFIG['text_chars']]}"


def queries(spec):
    """[variant][item] query texts; instruct encoders get the task line in front."""
    out = []
    for v in CONFIG['variants']:
        qs = [CONFIG['variant_templates'][v].format(item=t) for t in CONFIG['items'].values()]
        if spec['kind'] != 'cross':
            qs = [f"Instruct: {CONFIG['instruction']}\nQuery: {q}" for q in qs]
        out.append(qs)
    return out


def batched(fn, xs, bs, lens):
    """fn over xs in batches, longest first (an OOM shows up early), halving the batch on OOM."""
    import numpy as np, torch
    order = sorted(range(len(xs)), key=lambda i: -lens[i])
    res, i, t = [None] * len(xs), 0, time.time()
    while i < len(order):
        chunk = order[i:i + bs]
        try:
            out = fn([xs[j] for j in chunk])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            if bs == 1:
                raise
            bs = max(1, bs // 2); log('OOM -> batch', bs); continue
        for j, v in zip(chunk, out):
            res[j] = v
        i += len(chunk)
        if (i // len(chunk)) % 100 == 0:
            log(f'  {i}/{len(xs)} bs={bs} {time.time() - t:.0f}s')
    return np.stack(res)


def run_cross(spec):
    import numpy as np, torch
    from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
    # tokenizer.json directly: ru-reranker-edge's tokenizer_config names a transformers-5 class
    tok = PreTrainedTokenizerFast.from_pretrained(spec['name'])
    m = AutoModelForSequenceClassification.from_pretrained(spec['name'], torch_dtype=torch.float16).cuda().eval()

    def fn(b):
        x = tok([q for q, _ in b], [d for _, d in b], padding=True, truncation='only_second', max_length=512,
                return_tensors='pt').to('cuda')
        with torch.no_grad():
            return m(**x).logits[:, 0].float().cpu().numpy()

    docs = [n['text'] for n in CONFIG['news']]
    Q = queries(spec)
    pairs = [(q, d) for d in docs for qs in zip(*Q) for q in qs]  # order: news, item, variant
    lens = [len(q) + len(d) for q, d in pairs]
    raw = batched(fn, pairs, spec.get('bs', 64), lens).astype(np.float32).reshape(len(docs), len(CONFIG['items']), len(Q))
    return raw, 1 / (1 + np.exp(-raw))


def run_dense(spec):
    """sentence-transformers bi-encoder."""
    import numpy as np, torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(spec['name'], trust_remote_code=True, device='cuda',
                            model_kwargs={'torch_dtype': getattr(torch, spec.get('dtype', 'float16'))})
    m.max_seq_length = 512
    enc = lambda xs: batched(lambda b: m.encode(b, batch_size=len(b), normalize_embeddings=True, convert_to_numpy=True,
                                                show_progress_bar=False), xs, spec.get('bs', 32), [len(x) for x in xs])
    return cosine(enc, spec)


def run_hf(spec):
    """Big LLM-based encoders: transformers + device_map='auto' over both T4s, mean pooling as the model card says."""
    import numpy as np, torch, torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec['name'], trust_remote_code=True)
    model = AutoModel.from_pretrained(spec['name'], trust_remote_code=True, torch_dtype=torch.float16, device_map='auto').eval()
    dev = model.get_input_embeddings().weight.device

    def fn(b):
        x = tok(b, padding=True, truncation=True, max_length=512, return_tensors='pt').to(dev)
        with torch.no_grad():
            h = model(**x).last_hidden_state
        mask = x['attention_mask'].unsqueeze(-1).to(h.dtype)
        return F.normalize(((h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)).float(), dim=-1).cpu().numpy()

    enc = lambda xs: batched(fn, xs, spec.get('bs', 8), [len(x) for x in xs])
    return cosine(enc, spec)


def cosine(enc, spec):
    import numpy as np
    d = enc([n['text'] for n in CONFIG['news']])
    q = np.stack([enc(qs) for qs in queries(spec)], axis=-1)  # item x dim x variant
    assert np.isfinite(d).all() and np.isfinite(q).all(), 'NaN/inf in embeddings (fp16 overflow?)'
    raw = np.einsum('nd,idv->niv', d, q).astype(np.float32)
    return raw, np.clip(raw, 0, 1)


RUN = {'cross': run_cross, 'dense': run_dense, 'hf': run_hf}

if __name__ == '__main__':
    import numpy as np
    if len(sys.argv) > 1:  # child: one model
        spec = next(s for s in CONFIG['models'] if s['slug'] == sys.argv[1])
        try:
            t = time.time()
            load_texts()
            raw, score = RUN[spec['kind']](spec)
            np.savez(f"{OUT}/{spec['slug']}.npz", keys=np.array([n['key'] for n in CONFIG['news']]),
                     items=np.array(list(CONFIG['items'])), variants=np.array(CONFIG['variants']), score=score, raw=raw)
            log('done', spec['slug'], score.shape, f'{time.time() - t:.0f}s')
        except Exception:
            log('FAILED', spec['slug']); traceback.print_exc()
        sys.exit(0)
    log('pip'); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *CONFIG['pip']], check=False)
    import torch
    log('gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    lanes = []
    for gpus, slugs in CONFIG['gpu_groups']:  # one sequential lane per GPU group
        cmd = ' && '.join(f'{sys.executable} {os.path.abspath(__file__)} {s}' for s in slugs)
        lanes.append(subprocess.Popen(cmd, shell=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus}))
    for p in lanes:
        p.wait()
    log('all lanes finished', os.listdir(OUT))
