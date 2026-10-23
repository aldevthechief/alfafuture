"""Pick 10 OKVED x 1 company, all with REVENUE_COMPANY >= 15bn (the case's own definition of "large business",
see case-notes.md). Unlike pick_companies.py (leader/top20/top70 by rank), this is the *target-user* golden set:
every company already qualifies on its own; the picks diversify revenue from just above the 15bn floor to the
top of the OKVED, and the 10 OKVED codes diversify business archetype (trade / production / construction /
transport / finance / IT), not just "biggest 10 companies overall" (that would be 10x oil/metals/retail trade).
python -X utf8 stage2_company_facts/golden_set/pick_companies_large.py  -> stage2_company_facts/golden_set/companies_large.json"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.data import load

# okved -> 'top' (biggest in the OKVED) | 'min' (smallest among those still >= 15bn) -> picks the other end of the
# spectrum from the better-known top-1 companies already profiled in insight-examples.md / company-first-trace.md.
PICKS = {
    '46.21': 'min',     # опт зерном — top1 (U_34911, 53.9bn) уже разобран вручную, берём почти пороговую
    '41.20.9': 'mid',   # прочее строительство — top1 (U_177581) уже разобран, берём середину таблицы
    '46.71.2': 'min',   # опт моторным топливом — top1 кода 46.71 (U_57706, 105.6bn) уже разобран, берём пороговую
    '35.11.1': 'top',   # генерация тепловой энергии — производитель, флагман
    '46.46': 'min',     # опт фармацевтикой — U_81081 уже разобран, берём пороговую (третью из трёх)
    '27.12': 'top',     # произв. электрораспределительной аппаратуры — единственная ≥15bn (= U_106783, уже разобран)
    '42.11': 'mid',     # дорожное строительство — другой тип подряда, чем 41.20.9 (госзаказ, инфраструктура)
    '49.41.2': 'top',   # грузоперевозки — единственная ≥15bn, транспорт/логистика
    '64.91.2': 'top',   # лизинг — единственная ≥15bn (= U_14366, уже разобран), финансовые услуги
    '62.01': 'top',     # разработка ПО — единственная ≥15bn, едва выше порога (15.83bn), IT
}

U, P, _, _ = load()
deg = P.PAYER.value_counts().add(P.BENEFICIARY.value_counts(), fill_value=0)
out = {}
for code, pick in PICKS.items():
    g = U[(U.okved == code) & (U.REVENUE_COMPANY >= 15e9)].sort_values('REVENUE_COMPANY', ascending=False)
    n = len(g)
    idx = {'top': 0, 'min': n - 1, 'mid': n // 2}[pick]
    row = g.iloc[idx]
    out[row.SYNTHETIC_U] = {'okved': code, 'okved_name': row.MAINOKVED_NAME, 'pick': pick,
                             'rank': idx + 1, 'of_15bn_plus': n, 'revenue_bn': round(row.REVENUE_COMPANY / 1e9, 3),
                             'edges': int(deg.get(row.SYNTHETIC_U, 0))}
Path(__file__).with_name('companies_large.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for u, v in out.items():
    print(u, v)
