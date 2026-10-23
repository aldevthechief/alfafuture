"""Full-corpus channel scores: the 5 rankers' `item` scores (out/rank_all/<slug>.npz, push.py --all) -> ensemble =
mean over models of the per-item percentile rank over all news (as in evaluate.py). Event score = max over its reports.
Out: out/channels_news.parquet (news_id × A1..H1), out/channels_events.parquet (event × A1..H1).
python -X utf8 experiments/news_channels/ensemble.py"""
import numpy as np
import pandas as pd

from channels import DATA, ITEMS, OUT

IDS = list(ITEMS)


def model_scores(folder=OUT / 'rank_all'):
    out = {}
    for f in sorted(folder.glob('*.npz')):
        z = np.load(f)
        assert list(z['items']) == IDS and list(z['variants']) == ['item'], f
        out[f.stem] = pd.DataFrame(z['score'][:, :, 0], index=z['keys'].astype(str), columns=IDS)
    return out


if __name__ == '__main__':
    M = model_scores()
    print('models:', list(M), {k: v.shape for k, v in M.items()})
    assert len(M) == 5 and len({tuple(v.index) for v in M.values()}) == 1, 'need all 5 models over the same news'
    E = sum(S.rank(pct=True) for S in M.values()) / len(M)
    E.rename_axis('news_id').reset_index().to_parquet(OUT / 'channels_news.parquet', index=False)
    ev = pd.read_parquet(DATA / 'news' / 'events.parquet')[['event', 'news_ids']].explode('news_ids')
    ev['news_ids'] = ev['news_ids'].astype(str)
    V = ev.join(E, on='news_ids').dropna().groupby('event')[IDS].max()
    V.reset_index().to_parquet(OUT / 'channels_events.parquet', index=False)
    print(f'{len(E)} news, {len(V)} events; top item share per news:', E.idxmax(axis=1).value_counts(normalize=True).round(3).to_dict())

    # self-check: same news, same model -> same raw score as in the pilot kernel
    P = np.load(OUT / 'rank' / 'e5-large-instruct.npz')
    from judge import event_news
    t = event_news([int(k) for k in P['keys'][:20]])
    ids = [t[int(k)]['id'] for k in P['keys'][:20]]
    d = np.abs(M['e5-large-instruct'].loc[ids].values - P['score'][:20, :, 0]).max()
    print('pilot vs full e5 max |diff| on 20 news:', round(float(d), 4))
