"""Insight card as a standalone HTML page: what the Alfa-Business feed widget shows for one analyst-loop result.

- "Позиция Банка" and "Вывод AI" are separate blocks with their own colour and label; with no bank analytics the card
  carries the mandatory badge "вывод AI, не официальная позиция Банка";
- every claim has small source markers; hover or keyboard focus opens a popover: news title, outlet, date, link; or the
  bank report, date, page, freshness and the fragment; or a computed figure with its formula and source file;
- claims that failed the audit are marked "не подтверждено", hypotheses have their own block;
- impact level with its criteria and the code's estimate; "Шаги агента" unfolds the JSONL journal of the run;
- the "не инвестиционная рекомендация" line is always there.
Idea after FinRobot (desktop/src/components/SourcedNumber.tsx); our own code, no JS, CSS popovers only.
python -X utf8 stage6_insight_generation/insight_card/render.py result.json [--trace data/traces/<run>.jsonl] -o card.html
"""
import argparse, html, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CSS = """
:root{--red:#ef3124;--ink:#0b1f35;--muted:#5c6a79;--line:#e3e6ea;--bank:#fff4f3;--ai:#f3f5f8;--warn:#b25e00;--bg:#f7f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);font:15px/1.5 -apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:var(--ink)}
.card{max-width:760px;margin:24px auto;background:#fff;border-radius:16px;box-shadow:0 2px 12px rgba(11,31,53,.08);padding:24px}
@media(max-width:800px){.card{margin:0;border-radius:0;padding:16px}}
h1{font-size:21px;margin:0 0 6px}.meta{color:var(--muted);font-size:13px;margin-bottom:14px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0 16px}.chip{border-radius:999px;padding:3px 10px;font-size:12px;background:var(--ai)}
.chip.high{background:#fde2df;color:#a3150b}.chip.mid{background:#fff1d6;color:var(--warn)}.chip.low{background:#e7f4ea;color:#1d6b34}
.chip.ai{background:#e8ecf2;color:var(--ink)}
section{border-radius:12px;padding:12px 14px;margin:12px 0}section h2{font-size:13px;letter-spacing:.04em;text-transform:uppercase;margin:0 0 6px}
.bank{background:var(--bank);border-left:4px solid var(--red)}.bank h2{color:var(--red)}
.ai{background:var(--ai);border-left:4px solid #8a97a8}.facts{border:1px solid var(--line)}.hyp{border:1px dashed #b7c0cc}
.label{font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0}
ul{margin:0;padding-left:18px}li{margin:4px 0}
.unc{color:var(--warn);font-size:12px;border:1px solid var(--warn);border-radius:6px;padding:0 5px;margin-left:4px}
.ref{position:relative;display:inline-block;margin-left:2px}
.ref>button{all:unset;cursor:help;font-size:11px;color:var(--red);border:1px solid #f5b8b3;border-radius:6px;padding:0 4px;line-height:16px}
.ref>button:focus-visible{outline:2px solid var(--red)}
.pop{display:none;position:absolute;z-index:5;left:0;top:20px;width:min(340px,80vw);background:#fff;border:1px solid var(--line);
border-radius:10px;box-shadow:0 6px 24px rgba(11,31,53,.18);padding:10px 12px;font-size:13px;color:var(--ink);white-space:normal}
.ref:hover .pop,.ref:focus-within .pop{display:block}.pop b{display:block;margin-bottom:4px}.pop small{color:var(--muted);display:block}
.pop q{display:block;margin-top:6px;color:var(--muted);quotes:none}
.actions li{margin:6px 0}.disc{color:var(--muted);font-size:12px;margin-top:18px;border-top:1px solid var(--line);padding-top:10px}
details{margin-top:14px}summary{cursor:pointer;color:var(--red)}.step{font-size:12px;border-left:2px solid var(--line);padding:2px 0 6px 10px}
.step b{font-family:ui-monospace,Menlo,monospace}.warn{color:var(--warn)}
"""
LEVEL = {'high': 'высокое влияние', 'mid': 'среднее влияние', 'low': 'низкое влияние'}
CONF = {'high': 'уверенность высокая', 'medium': 'уверенность средняя', 'low': 'уверенность низкая'}
STEP = {'start': 'старт', 'event': 'разбор события', 'bank_match': 'подбор аналитики Банка', 'enrich': 'возврат к пулу новостей',
        'compute': 'расчёт последствий', 'risks': 'риски и возможности', 'assess': 'объяснение моделью', 'grounds': 'проверка оснований',
        'request': 'запрос данных', 'request_open': 'запрос не закрыт', 'verdict': 'вердикт', 'write': 'формулировка инсайта',
        'audit': 'проверка ответа', 'end': 'готово', 'injection_suspected': 'подозрение на инъекцию'}
e = html.escape


class Sources:
    def __init__(self, result: dict):
        b = result.get('verdict', {}).get('brief') or {}
        self.facts = {f['id']: f for f in b.get('facts', []) + b.get('hypotheses', [])}
        self.bank = {x['id']: x for x in result.get('bank', [])}
        ev = result.get('event', {})
        self.news = {ev.get('id'): ev} if ev.get('origin', 'news') == 'news' else {}
        for r in result.get('news', []):
            self.news[f'news:{r["id"]}'] = r

    def popover(self, ref: str) -> str:
        if ref in self.news:
            n = self.news[ref]
            link = f'<small><a href="{e(n.get("url") or "#")}" target="_blank" rel="noopener">{e(n.get("url") or "ссылки нет")}</a></small>'
            return f'<b>Новость</b>{e(n.get("title", ""))}<small>{e(str(n.get("source", "")))} · {e(str(n.get("date") or "дата неизвестна"))}</small>{link}'
        bank = self.bank.get(ref) or next((v for k, v in self.bank.items() if k.startswith(ref)), None)
        if bank:
            shaky = '<small>значения по шкале графика — приблизительно, не проверено</small>' if bank.get('unverified') else ''
            return (f'<b>Аналитика Банка</b>{e(bank.get("label") or bank["report"])}<small>{e(str(bank["date"]))} · стр. {bank["page"]} · '
                    f'{e(bank["freshness"])}{" · позиция Банка" if bank.get("own_view") else ""}</small>{shaky}'
                    f'<q>{e(bank["text"][:300])}</q>')
        if ref in self.facts:
            f = self.facts[ref]
            tag = 'Допущение (гипотеза)' if f.get('kind') == 'hypothesis' else \
                'Факт о компании (этап 2)' if f.get('source') == 'stage2_company_facts' else 'Посчитано кодом'
            return (f'<b>{tag}</b>{e(f["label"])}: {e(str(f["value"]))} {e(f["unit"])}<small>формула: {e(f["formula"] or "—")}</small>'
                    f'<small>источник: {e(f["source"] or "—")}</small>')
        return f'<b>{e(ref)}</b><small>описания источника нет</small>'

    def marker(self, ref: str, n: int) -> str:
        return f'<span class="ref"><button aria-label="источник {e(ref)}">{n}</button><span class="pop" role="tooltip">{self.popover(ref)}</span></span>'


def claims(items: list[dict], src: Sources) -> str:
    rows = []
    for c in items:
        marks = ''.join(src.marker(r, i) for i, r in enumerate(c.get('sources', []), 1))
        unc = '<span class="unc">не подтверждено</span>' if c.get('unconfirmed') or not c.get('sources') else ''
        quotes = ''.join(f'<blockquote>{e(q["quote"])}<small>{e(q["source"])} · {q.get("start")}:{q.get("end")}</small></blockquote>'
                         for q in c.get('evidence', []))
        rows.append(f'<li>{e(c["text"])}{marks}{unc}{quotes}</li>')
    return '<ul>' + ''.join(rows) + '</ul>' if rows else '<p class="label">нет</p>'


def timeline(events: list[dict]) -> str:
    out = []
    for ev in events:
        d = ev.get('data', {})
        brief = ', '.join(f'{k}={v}' for k, v in d.items() if k not in ('facts', 'matches', 'issues', 'items', 'consequences', 'queries')
                          and not isinstance(v, (list, dict)))[:220]
        extra = ''
        if ev['step'] == 'bank_match':
            extra = '; '.join(f'{m["id"]} ({m["score"]})' for m in d.get('matches', []))
        if ev['step'] == 'grounds':
            extra = '; '.join(i['detail'] for i in d.get('issues', []))
        out.append(f'<div class="step"><b>{e(STEP.get(ev["step"], ev["step"]))}</b> {e(brief)}<br>{e(extra[:400])}</div>')
    return ''.join(out)


def render(result: dict, trace: list[dict] | None = None) -> str:
    src = Sources(result)
    ins = result.get('insight')
    verdict = result.get('verdict', {})
    if not ins:
        body = (f'<h1>Событие отклонено</h1><div class="meta">{e(result.get("title", ""))}</div><ul>'
                + ''.join(f'<li>{e(r)}</li>' for r in verdict.get('reasons', [])) + '</ul>')
    else:
        imp = ins.get('impact') or {}
        chips = ['<span class="chip">Из аналитики Банка</span>' if ins.get('origin') == 'bank' else '',
                 f'<span class="chip {imp.get("level", "")}" title="{e(imp.get("criteria", ""))}">{LEVEL.get(imp.get("level"), "")}'
                 f'{": " + e(imp["estimate"]) if imp.get("estimate") else ""}</span>' if imp else '',
                 f'<span class="chip">{CONF.get(ins.get("confidence"), "")}</span>',
                 '<span class="chip ai">вывод AI, не официальная позиция Банка</span>' if ins.get('not_bank_position') else '']
        bank = '' if ins.get('not_bank_position') else \
            f'<section class="bank"><h2>Позиция Банка <span class="label">— аналитика Альфа-Банка</span></h2>{claims(ins["bank_view"], src)}</section>'
        ai = (f'<section class="ai"><h2>Вывод AI <span class="label">— сгенерировано моделью, проверено кодом</span></h2>'
              f'<p><b>Почему это важно</b></p>{claims(ins["why_it_matters"], src)}<p><b>Риски</b></p>{claims(ins["risks"], src)}'
              f'<p><b>Возможности</b></p>{claims(ins["opportunities"], src)}<p>{e(ins.get("ai_conclusion", ""))}</p></section>')
        hyp = f'<section class="hyp"><h2>Гипотезы <span class="label">— допущения, не подтверждены данными</span></h2>{claims(ins["hypotheses"], src)}</section>' \
            if ins.get('hypotheses') else ''
        product = f'<p class="label">Продукт Банка по задаче: {e(ins["bank_product"])}</p>' if ins.get('bank_product') else ''
        source_label = 'Тезис Банка' if ins.get('origin') == 'bank' else 'Что произошло'
        body = (f'<h1>{e(ins["title"])}</h1><div class="meta">{e(result.get("title", ""))} · роль: {e(ins.get("role", ""))}</div>'
                f'<div class="chips">{"".join(chips)}</div>'
                f'<section class="facts"><h2>{source_label} <span class="label">— факты из источников</span></h2>{claims(ins["what_happened"], src)}</section>'
                f'{bank}{ai}{hyp}<section class="facts"><h2>Что сделать</h2><ul class="actions">'
                + ''.join(f'<li>{e(a)}</li>' for a in ins.get('actions', [])) + f'</ul>{product}</section>')
        warns = [w for w in result.get('audit', []) if w['code'] in ('unknown_ref', 'no_source', 'moved_to_ai')]
        if warns:
            body += '<p class="warn">Проверка ответа: ' + e('; '.join(f'{w["code"]} — {w["text"][:80]}' for w in warns)) + '</p>'
    steps = f'<details><summary>Шаги агента ({len(trace)})</summary>{timeline(trace)}</details>' if trace else ''
    disc = (ins or {}).get('disclaimer') or 'Не является инвестиционной рекомендацией.'
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Инсайт для бизнеса</title><style>{CSS}</style></head><body><article class="card">{body}{steps}'
            f'<p class="disc">{e(disc)} Материал носит информационный характер.</p></article></body></html>')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('result')
    ap.add_argument('--trace', default=None)
    ap.add_argument('-o', '--out', default='card.html')
    a = ap.parse_args()
    from common.trace import AgentTrace
    result = json.loads(Path(a.result).read_text(encoding='utf-8'))
    Path(a.out).write_text(render(result, AgentTrace.read(Path(a.trace)) if a.trace else None), encoding='utf-8')
    print(a.out)
