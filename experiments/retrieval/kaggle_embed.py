"""Kaggle kernel body: embed the news pool + synthetic needles and the generator queries with several encoders.
CONFIG (models, queries, needles, instructions) is injected by push.py. Writes /kaggle/working/<slug>.npz per model.
Each model runs in its own subprocess pinned to its GPU group; a failing model is logged and skipped."""
import gc, json, os, subprocess, sys, time, traceback

CONFIG = json.loads(r'''__CONFIG__''')
OUT = '/kaggle/working'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def docs():
    import glob
    import pandas as pd
    path = (glob.glob('/kaggle/input/**/news.parquet', recursive=True) or [CONFIG['news_path']])[0]  # mount path varies
    df = pd.read_parquet(path)
    texts = (df.title.fillna('') + '. ' + df.title.fillna('') + '. ' + df.text.fillna('').str[:2000]).tolist()
    ids = df['id'].astype(str).tolist()
    for n in CONFIG['needles']:
        ids.append(n['id'])
        texts.append(f"{n['title']}. {n['title']}. {n['text']}")
    return ids, texts


def query_texts(spec):
    """{variant: [formatted query per CONFIG['queries'] item]}"""
    fam, out = spec['family'], {}
    Q = CONFIG['queries']
    if fam in ('instruct', 'instruct_nospace'):
        sep = 'Query:' if fam == 'instruct_nospace' else 'Query: '
        for v, task in CONFIG['instructions'].items():
            out[v] = [f"Instruct: {task}\n{sep}{q['text']}" for q in Q]
        out['reason'] = [f"Instruct: {CONFIG['reason_instructions'][q['reason']]}\n{sep}{q['query']}" for q in Q]
        out['none'] = [q['text'] for q in Q]
    else:
        p = spec.get('qprefix', '')
        out['text'] = [p + q['text'] for q in Q]
        out['query'] = [p + q['query'] for q in Q]
    return out


def encode_dense(model, texts, bs, **kw):
    import numpy as np, torch
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # long first: an OOM shows up early
    res, i = [None] * len(texts), 0
    while i < len(order):
        chunk = order[i:i + bs]
        try:
            e = model.encode([texts[j] for j in chunk], batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                             show_progress_bar=False, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            if bs == 1:
                raise
            bs = max(1, bs // 2); log('OOM -> batch', bs); continue
        for j, v in zip(chunk, e):
            res[j] = v
        i += len(chunk)
        if i // len(chunk) % 50 == 0:
            log(f'  {i}/{len(texts)} bs={bs}')
    return np.stack(res).astype(np.float16)


def run_dense(spec):
    import numpy as np, torch
    from sentence_transformers import SentenceTransformer
    mk = {'torch_dtype': torch.float16}
    if spec.get('device_map'):
        mk['device_map'] = 'auto'
    m = SentenceTransformer(spec['name'], trust_remote_code=True, model_kwargs=mk,
                            tokenizer_kwargs={'padding_side': spec['padding_side']} if spec.get('padding_side') else None,
                            device=None if spec.get('device_map') else 'cuda')
    m.max_seq_length = spec.get('max_len', 512)
    ids, texts = docs()
    t = time.time()
    d = encode_dense(m, [spec.get('dprefix', '') + x for x in texts], spec.get('bs', 16))
    log(spec['slug'], 'docs', d.shape, f'{time.time() - t:.0f}s')
    qs = {f'q_{v}': encode_dense(m, txt, 32) for v, txt in query_texts(spec).items()}
    np.savez(f"{OUT}/{spec['slug']}.npz", doc_ids=np.array(ids), doc=d, **qs)


def run_sparse(spec):
    import numpy as np, scipy.sparse as sp
    from sentence_transformers import SparseEncoder
    import torch
    m = SparseEncoder(spec['name'], device='cuda', trust_remote_code=True)
    m.max_seq_length = 512
    ids, texts = docs()

    def to_csr(x):
        x = x.coalesce().cpu()
        i = x.indices().numpy()
        return sp.csr_matrix((x.values().float().numpy(), (i[0], i[1])), shape=tuple(x.shape))

    def enc(fn, xs, bs):  # vocab-sized logits (bs x 512 x 250k) are the memory hog: chunk and halve on OOM
        parts, i = [], 0
        while i < len(xs):
            try:
                parts.append(to_csr(fn(xs[i:i + 64], batch_size=bs, convert_to_sparse_tensor=True)))
                i += 64
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                if bs == 1:
                    raise
                bs //= 2; log('OOM -> batch', bs)
        return sp.vstack(parts).tocsr()

    t = time.time()
    d = enc(m.encode_document, texts, spec.get('bs', 4))
    log(spec['slug'], 'docs', d.shape, f'{time.time() - t:.0f}s')
    sp.save_npz(f"{OUT}/{spec['slug']}_doc.npz", d)
    for v, txt in query_texts(spec).items():
        sp.save_npz(f"{OUT}/{spec['slug']}_q_{v}.npz", enc(m.encode_query, txt, 16))
    np.save(f"{OUT}/{spec['slug']}_doc_ids.npy", np.array(ids))


def run_bge_m3(spec):
    """BGE-M3 in one pass: dense + learned lexical weights (its sparse head)."""
    import numpy as np, scipy.sparse as sp
    from FlagEmbedding import BGEM3FlagModel
    m = BGEM3FlagModel(spec['name'], use_fp16=True, devices=['cuda:0'])
    ids, texts = docs()

    def enc(xs, bs):
        o = m.encode(xs, batch_size=bs, max_length=512, return_dense=True, return_sparse=True)
        rows = [(i, int(k), float(w)) for i, lw in enumerate(o['lexical_weights']) for k, w in lw.items()]
        r, c, w = zip(*rows) if rows else ((), (), ())
        return o['dense_vecs'].astype(np.float16), sp.csr_matrix((w, (r, c)), shape=(len(xs), 250002))

    t = time.time()
    dd, ds = enc(texts, spec.get('bs', 16))
    log(spec['slug'], 'docs', dd.shape, f'{time.time() - t:.0f}s')
    qd = {}
    for v, txt in query_texts(spec).items():
        a, b = enc(txt, 64)
        qd[f'q_{v}'] = a
        sp.save_npz(f"{OUT}/{spec['slug']}-sparse_q_{v}.npz", b)
    sp.save_npz(f"{OUT}/{spec['slug']}-sparse_doc.npz", ds)
    np.save(f"{OUT}/{spec['slug']}-sparse_doc_ids.npy", np.array(ids))
    np.savez(f"{OUT}/{spec['slug']}.npz", doc_ids=np.array(ids), doc=dd, **qd)


def run_hf(spec):
    """Big LLM-based encoders: transformers + device_map='auto' over both T4s, pooling as the model card says."""
    import numpy as np, torch, torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec['name'], trust_remote_code=True, padding_side=spec.get('padding_side', 'right'))
    model = AutoModel.from_pretrained(spec['name'], trust_remote_code=True, torch_dtype=torch.float16, device_map='auto').eval()
    dev = model.get_input_embeddings().weight.device

    def pool(h, mask):
        if spec['pooling'] == 'last':  # left padding: the last position is the last real token
            return h[:, -1]
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def enc(texts, bs):
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        res, i = [None] * len(texts), 0
        while i < len(order):
            chunk = order[i:i + bs]
            try:
                b = tok([texts[j] for j in chunk], padding=True, truncation=True, max_length=spec.get('max_len', 512), return_tensors='pt').to(dev)
                with torch.no_grad():
                    e = F.normalize(pool(model(**b).last_hidden_state, b['attention_mask']).float(), dim=-1).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); gc.collect()
                if bs == 1:
                    raise
                bs = max(1, bs // 2); log('OOM -> batch', bs); continue
            for j, v in zip(chunk, e):
                res[j] = v
            i += len(chunk)
            if (i // len(chunk)) % 100 == 0:
                log(f'  {i}/{len(texts)} bs={bs}')
        out = np.stack(res)
        assert np.isfinite(out).all(), 'NaN/inf in embeddings (fp16 overflow?)'
        return out.astype(np.float16)

    ids, texts = docs()
    t = time.time()
    d = enc(texts, spec.get('bs', 8))
    log(spec['slug'], 'docs', d.shape, f'{time.time() - t:.0f}s')
    qs = {f'q_{v}': enc(txt, 32) for v, txt in query_texts(spec).items()}
    np.savez(f"{OUT}/{spec['slug']}.npz", doc_ids=np.array(ids), doc=d, **qs)


RUN = {'dense': run_dense, 'sparse': run_sparse, 'bge_m3': run_bge_m3, 'hf': run_hf}

if __name__ == '__main__':
    if len(sys.argv) > 1:  # child: one model
        spec = next(s for s in CONFIG['models'] if s['slug'] == sys.argv[1])
        try:
            RUN[spec['kind']](spec)
            log('done', spec['slug'])
        except Exception:
            log('FAILED', spec['slug']); traceback.print_exc()
        sys.exit(0)
    log('pip'); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *CONFIG['pip']], check=False)
    import torch
    log('gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    groups = CONFIG['gpu_groups']  # e.g. [["frida", "e5"], ["user", "bge"]] -> one sequential lane per GPU; [[...]] with "0,1" for device_map
    lanes = []
    for gpus, slugs in groups:
        cmd = ' && '.join(f'{sys.executable} {os.path.abspath(__file__)} {s}' for s in slugs)
        lanes.append(subprocess.Popen(cmd, shell=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus}))
    for p in lanes:
        p.wait()
    log('all lanes finished', os.listdir(OUT))
