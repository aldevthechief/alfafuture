"""Pick 3 OKVED x 3 companies for the golden set: a leader (top-3), a solid one (~top-20%), a below-median one (~top-70%).
Within each band the company with the most payment edges is taken (more evidence to analyse).
python -X utf8 stage2_company_facts/golden_set/pick_companies.py  -> stage2_company_facts/golden_set/companies.json"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import load

OKVEDS = ['41.20', '62.01', '56.10']  # construction / software / restaurants: common and unlike each other
BANDS = {'leader': (0, 3), 'top20': (0.17, 0.23), 'top70': (0.67, 0.73)}  # rank share bounds

U, P, _, _ = load()
deg = P.PAYER.value_counts().add(P.BENEFICIARY.value_counts(), fill_value=0)
out = {}
for code in OKVEDS:
    g = U[U.okved == code].sort_values('REVENUE_COMPANY', ascending=False).reset_index(drop=True)
    n = len(g)
    for band, (lo, hi) in BANDS.items():
        sl = g.iloc[lo:hi] if band == 'leader' else g.iloc[int(lo * n):int(hi * n)]
        best = sl.assign(deg=sl.SYNTHETIC_U.map(deg).fillna(0)).sort_values('deg', ascending=False).iloc[0]
        rank = int(g.index[g.SYNTHETIC_U == best.SYNTHETIC_U][0]) + 1
        out[best.SYNTHETIC_U] = {'okved': code, 'band': band, 'rank': rank, 'of': n, 'edges': int(best.deg),
                                 'revenue_bn': round(best.REVENUE_COMPANY / 1e9, 3)}
Path(__file__).with_name('companies.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for u, v in out.items():
    print(u, v)
