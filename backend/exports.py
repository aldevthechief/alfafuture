"""Exports of insights, payments and counterparties: tables built from saved results and the case tables, no LLM calls.

A table is (columns, rows): columns are (key, Russian title) pairs; JSON keeps the keys (stable for Power BI / DataLens /
pandas), CSV and XLSX show the titles. Insights keep the case's layers apart: fact / bank analytics / AI conclusion /
hypothesis per claim. From the bank's reports only what the card shows leaves the bank: the bank_view claims with a short
quote, report title, date and page; never fragments, OCR, chart digitisation or the PDFs."""
import csv
import io
import json
import zipfile
from datetime import datetime, timezone

QUOTE_CHARS = 300
LAYER = {'fact': 'Факт', 'bank': 'Аналитика Банка', 'ai': 'Вывод AI', 'hypothesis': 'Гипотеза'}
SECTION = {'what_happened': 'Что случилось', 'why_it_matters': 'Почему важно', 'risks': 'Риск',
           'opportunities': 'Возможность', 'bank_view': 'Позиция Банка', 'hypotheses': 'Гипотеза'}
NOT_BANK = 'Не является официальной позицией Банка.'
NOT_ADVICE = 'Не является инвестиционной рекомендацией.'

INSIGHTS = [('insight_id', 'ID инсайта'), ('company_id', 'Компания'), ('role', 'Роль'), ('title', 'Заголовок'),
            ('origin', 'Источник события'), ('kind', 'Тип'), ('event_date', 'Дата события'), ('channel', 'Канал'),
            ('impact', 'Влияние'), ('impact_estimate', 'Оценка влияния'), ('confidence', 'Уверенность'),
            ('ai_conclusion', 'Вывод AI'), ('bank_product', 'Продукт Банка'), ('claims', 'Утверждений'),
            ('unconfirmed', 'Не подтверждено'), ('not_bank_position', 'Без позиции Банка'), ('disclaimer', 'Оговорка'),
            ('job_id', 'Задание'), ('bank_version', 'Версия аналитики Банка')]
CLAIMS = [('insight_id', 'ID инсайта'), ('claim_id', 'ID утверждения'), ('section', 'Раздел'), ('layer', 'Слой'),
          ('text', 'Текст'), ('unconfirmed', 'Не подтверждено'), ('sources', 'Источники'), ('depends_on', 'Опирается на гипотезы'),
          ('check', 'Что подтвердит гипотезу')]
SOURCES = [('insight_id', 'ID инсайта'), ('claim_id', 'ID утверждения'), ('source', 'Источник'), ('type', 'Тип'),
           ('title', 'Название'), ('date', 'Дата'), ('page', 'Страница'), ('url', 'Ссылка'), ('quote', 'Цитата'),
           ('status', 'Проверка цитаты')]
ACTIONS = [('insight_id', 'ID инсайта'), ('n', '№'), ('text', 'Действие'), ('because', 'Потому что'),
           ('depends_on', 'Опирается на гипотезы')]
PAYMENTS = [('company_id', 'Компания'), ('direction', 'Направление'), ('counterparty_id', 'Контрагент'),
            ('amount', 'Сумма, ₽'), ('payments', 'Платежей'), ('share', 'Доля стороны'), ('same_group', 'Своя группа'),
            ('okved', 'ОКВЭД контрагента'), ('okved_name', 'Вид деятельности'), ('region', 'Регион'),
            ('alfa_client', 'Клиент Альфа-Банка'), ('in_table', 'Есть в справочнике')]
COUNTERPARTIES = [('company_id', 'Компания'), ('side', 'Сторона'), ('rank', 'Место'), ('counterparty_id', 'Контрагент'),
                  ('amount', 'Сумма, ₽'), ('payments', 'Платежей'), ('share', 'Доля стороны'), ('okved', 'ОКВЭД'),
                  ('okved_name', 'Вид деятельности'), ('region', 'Регион'), ('scale_class', 'Масштаб'),
                  ('same_group', 'Своя группа'), ('alfa_client', 'Клиент Альфа-Банка')]
CONCENTRATION = [('company_id', 'Компания'), ('side', 'Сторона'), ('total', 'Оборот стороны, ₽'), ('n', 'Контрагентов'),
                 ('top1_share', 'Доля крупнейшего'), ('top5_share', 'Доля топ-5'), ('hhi', 'HHI'),
                 ('dependence', 'Зависимость (HHI ≥ 0,25)')]
GROUP = [('company_id', 'Компания'), ('group_id', 'Группа'), ('size', 'Компаний в группе'), ('revenue_total', 'Выручка группы, ₽'),
         ('company_share', 'Доля компании'), ('company_rank', 'Место в группе'), ('alfa_clients', 'Клиентов Альфы'),
         ('inner_amount', 'Платежи внутри группы, ₽'), ('persons', 'Связанных лиц'),
         ('linked_companies', 'Компаний через общих лиц'), ('outside_group', 'Из них вне группы')]
PAYMENTS_NOTE = ('Агрегаты платежей за всю доступную выборку; период в исходных данных не указан '
                 '(сумма и число по паре плательщик-получатель), не банковская выписка. '
                 'Переводы между своими счетами — отдельной строкой «Между своими счетами». Контрагенты вне справочника '
                 'остаются только с идентификатором, без восстановления.')


def _text(claim):
    return claim.get('text', '') if isinstance(claim, dict) else str(claim)


def _short(text):
    text = ' '.join(str(text or '').split())
    return text if len(text) <= QUOTE_CHARS else text[:QUOTE_CHARS - 1] + '…'


def insight_tables(results, *, job=None, company_id=None, role=None):
    """Rows of a job result (or data/insights/<company>_<role>.json): only ready insights, as the feed shows them."""
    request = (job or {}).get('request') or {}
    company_id = company_id or request.get('company_id') or (job or {}).get('result', {}).get('company_id')
    bank_version = ((job or {}).get('result') or {}).get('bank_version') or (job or {}).get('bank_version')
    out = dict(insights=[], claims=[], sources=[], actions=[])
    for index, row in enumerate(results or []):
        insight = row.get('insight')
        if row.get('status') != 'ready' or not insight:
            continue
        iid = insight.get('id') or f'row-{index}'
        claims = [(section, c) for section in SECTION for c in insight.get(section) or []]
        impact = insight.get('impact') or {}
        bank = row.get('source') == 'bank_analytics' or insight.get('origin') == 'bank'
        out['insights'].append(dict(
            insight_id=iid, company_id=company_id or insight.get('company_id'), role=role or request.get('role') or insight.get('role'),
            title=insight.get('title', ''), origin='Аналитика Банка' if bank else 'Новость', kind=insight.get('kind', ''),
            event_date=insight.get('event_date', ''), channel=insight.get('channel', ''), impact=impact.get('level', ''),
            impact_estimate=impact.get('estimate', ''), confidence=insight.get('confidence', ''),
            ai_conclusion=insight.get('ai_conclusion', ''), bank_product=insight.get('bank_product', ''), claims=len(claims),
            unconfirmed=sum(bool(isinstance(c, dict) and c.get('unconfirmed')) for _, c in claims),
            not_bank_position=bool(insight.get('not_bank_position') or not insight.get('bank_view')),
            disclaimer=' '.join(x for x in (insight.get('disclaimer') or NOT_ADVICE,
                                            NOT_BANK if insight.get('not_bank_position') or not insight.get('bank_view') else '') if x),
            job_id=(job or {}).get('id', ''), bank_version=bank_version or ''))
        for n, (section, claim) in enumerate(claims, 1):
            claim = claim if isinstance(claim, dict) else dict(text=str(claim))
            cid = claim.get('id') or f'{section}:{n}'
            layer = 'bank' if section == 'bank_view' else 'hypothesis' if section == 'hypotheses' else claim.get('kind', 'ai')
            out['claims'].append(dict(insight_id=iid, claim_id=cid, section=SECTION[section], layer=LAYER.get(layer, layer),
                                      text=_text(claim), unconfirmed=bool(claim.get('unconfirmed')),
                                      sources=', '.join(claim.get('sources') or []), depends_on=', '.join(claim.get('depends_on') or []),
                                      check=claim.get('check', '')))
            for e in claim.get('evidence') or []:
                source = e.get('source', '')
                kind = 'Отчёт Банка' if source.startswith('bank:') else 'Новость' if source.startswith('news:') else 'Данные компании'
                out['sources'].append(dict(insight_id=iid, claim_id=cid, source=source, type=kind, title=e.get('title', ''),
                                           date=e.get('date', ''), page=e.get('page'), url=e.get('url', ''),
                                           quote=_short(e.get('quote')), status=e.get('status', '')))
        for ref, news in (row.get('news_references') or {}).items():
            if isinstance(news, dict):
                out['sources'].append(dict(insight_id=iid, claim_id='', source=ref, type='Новость', title=news.get('title', ''),
                                           date=news.get('date', ''), page=None, url=news.get('url', ''), quote='', status=''))
        details = insight.get('action_details') or []
        for n, action in enumerate(details or [dict(text=a) for a in insight.get('actions') or []], 1):
            out['actions'].append(dict(insight_id=iid, n=n, text=action.get('text', ''), because=action.get('because', ''),
                                       depends_on=', '.join(action.get('depends_on') or [])))
    return dict(insights=(INSIGHTS, out['insights']), claims=(CLAIMS, out['claims']), sources=(SOURCES, out['sources']),
                actions=(ACTIONS, out['actions']))


def group_members(U, company_id):
    return list(U.index[U.GROUP_ID == U.at[company_id, 'GROUP_ID']])


def payment_tables(U, P, S, company_ids, side='both'):
    """All counterparties of the companies (both directions), plus their self-transfers as their own row."""
    rows = []
    for u in company_ids:
        group = U.at[u, 'GROUP_ID']
        for direction, me, cp in (('out', 'PAYER', 'BENEFICIARY'), ('in', 'BENEFICIARY', 'PAYER')):
            if side not in ('both', direction):
                continue
            g = P[P[me] == u].groupby(cp).agg(amount=('PAY_AMT', 'sum'), count=('PAY_CNT', 'sum')).sort_values('amount', ascending=False)
            total = float(g.amount.sum())
            for x, r in g.iterrows():
                known = x in U.index
                c = U.loc[x] if known else None
                rows.append(dict(company_id=u, direction='Исходящие' if direction == 'out' else 'Входящие', counterparty_id=x,
                                 amount=float(r.amount), payments=int(r['count']), share=round(float(r.amount / total), 4) if total else None,
                                 same_group=bool(known and c.GROUP_ID == group), okved=c.okved if known else '',
                                 okved_name=c.MAINOKVED_NAME if known else '', region=c.REGION if known else '',
                                 alfa_client=bool(known and c.CL_TYPE == 'CLIENT'), in_table=known))
        if u in S.index:
            own = S.loc[[u]]
            rows.append(dict(company_id=u, direction='Между своими счетами', counterparty_id=u, amount=float(own.PAY_AMT.sum()),
                             payments=int(own.PAY_CNT.sum()), share=None, same_group=True, okved=U.at[u, 'okved'],
                             okved_name=U.at[u, 'MAINOKVED_NAME'], region=U.at[u, 'REGION'],
                             alfa_client=bool(U.at[u, 'CL_TYPE'] == 'CLIENT'), in_table=True))
    return dict(payments=(PAYMENTS, rows))


def counterparty_tables(U, P, F, company_ids, top=10):
    from stage2_company_facts.links.links import links
    tops, conc, groups = [], [], []
    for u in company_ids:
        found = links(U, P, F, u, k=top)
        for name, title in (('suppliers', 'Поставщики (исходящие платежи)'), ('buyers', 'Покупатели (входящие платежи)')):
            s = found[name]
            conc.append(dict(company_id=u, side=title, total=s['total'], n=s['n'], top1_share=s['top1_share'],
                             top5_share=s['top5_share'], hhi=s['hhi'], dependence=bool(s['hhi'] is not None and s['hhi'] >= 0.25)))
            for rank, item in enumerate(s['top'], 1):
                tops.append(dict(company_id=u, side=title, rank=rank, counterparty_id=item['id'], amount=item['amount'],
                                 payments=item['payments'], share=item['share'], okved=item.get('okved', ''),
                                 okved_name=item.get('okved_name', ''), region=item.get('region', ''),
                                 scale_class=item.get('scale_class', ''), same_group=item.get('same_group'),
                                 alfa_client=item.get('alfa_client')))
        g, persons = found['group'] or {}, found['persons'] or {}
        groups.append(dict(company_id=u, group_id=g.get('id') or U.at[u, 'GROUP_ID'], size=g.get('size', 1),
                           revenue_total=g.get('revenue_total'), company_share=g.get('company_share'), company_rank=g.get('company_rank'),
                           alfa_clients=g.get('alfa_clients'), inner_amount=(g.get('inner_payments') or {}).get('amount'),
                           persons=persons.get('persons', 0), linked_companies=persons.get('linked_companies', 0),
                           outside_group=persons.get('outside_group', 0)))
    return dict(counterparties=(COUNTERPARTIES, tops), concentration=(CONCENTRATION, conc), group=(GROUP, groups))


# --- writers

SHEETS = dict(insights='Инсайты', claims='Утверждения', sources='Источники', actions='Действия', payments='Платежи',
              counterparties='Контрагенты', concentration='Концентрация', group='Группа')
MEDIA = dict(csv='text/csv; charset=utf-8', json='application/json', jsonl='application/x-ndjson',
             xlsx='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', zip='application/zip')


def _cell_csv(value, decimal):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if isinstance(value, float):
        text = repr(round(value, 6)).rstrip('0').rstrip('.') if value != int(value) else str(int(value))
        return text.replace('.', ',') if decimal == 'comma' else text
    text = str(value)
    # CSV injection: a news title like «=HYPERLINK(...)» must stay text in the reader's spreadsheet
    return "'" + text if text[:1] in ('=', '+', '-', '@', '\t', '\r') else text


def to_csv(table, decimal='comma'):
    columns, rows = table
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=';', lineterminator='\r\n')
    writer.writerow([title for _, title in columns])
    for row in rows:
        writer.writerow([_cell_csv(row.get(key), decimal) for key, _ in columns])
    return ('﻿' + buffer.getvalue()).encode('utf-8')  # BOM: Russian Excel opens UTF-8 without the import wizard


def to_json(tables, meta):
    return json.dumps(dict(meta=meta, tables={name: rows for name, (_, rows) in tables.items()}), ensure_ascii=False,
                      default=str, indent=1).encode('utf-8')


def to_jsonl(table):
    return ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in table[1]).encode('utf-8')


def to_zip(tables, meta, decimal='comma'):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, table in tables.items():
            z.writestr(f'{name}.csv', to_csv(table, decimal))
        z.writestr('meta.json', json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    return buffer.getvalue()


def to_xlsx(tables, meta):
    import xlsxwriter
    buffer = io.BytesIO()
    # strings_to_formulas/urls off: every value stays a plain cell, nothing from a news text becomes a formula or link
    book = xlsxwriter.Workbook(buffer, {'in_memory': True, 'strings_to_formulas': False, 'strings_to_urls': False,
                                        'strings_to_numbers': False})
    head = book.add_format({'bold': True, 'bg_color': '#EF3124', 'font_color': '#FFFFFF', 'text_wrap': True, 'valign': 'top'})
    wrap = book.add_format({'text_wrap': True, 'valign': 'top'})
    money = book.add_format({'num_format': '#,##0', 'valign': 'top'})
    share = book.add_format({'num_format': '0.0%', 'valign': 'top'})
    for name, (columns, rows) in tables.items():
        sheet = book.add_worksheet(SHEETS.get(name, name)[:31])
        for j, (key, title) in enumerate(columns):
            sheet.write(0, j, title, head)
            longest = max([len(title)] + [len(str(r.get(key) or '')) for r in rows[:200]])
            style = money if key in ('amount', 'total', 'revenue_total', 'inner_amount') else \
                share if key.endswith('share') else wrap
            sheet.set_column(j, j, min(max(10, longest + 2), 60), style)
        for i, row in enumerate(rows, 1):
            for j, (key, _) in enumerate(columns):
                value = row.get(key)
                if isinstance(value, bool):
                    value = 'да' if value else 'нет'
                if value is None or value == '':
                    continue
                sheet.write(i, j, value)
        sheet.freeze_panes(1, 0)
        if columns:
            sheet.autofilter(0, 0, max(len(rows), 1), len(columns) - 1)
    about = book.add_worksheet('О выгрузке')
    about.set_column(0, 0, 34); about.set_column(1, 1, 100, wrap)
    for i, (key, value) in enumerate(meta.items()):
        about.write(i, 0, key, head)
        about.write(i, 1, json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value))
    book.close()
    return buffer.getvalue()


def render(tables, meta, fmt, table=None, decimal='comma'):
    """-> (bytes, media type, file extension). CSV and JSON Lines hold one table (`table`, default the first); several
    tables in CSV go as a zip of CSV files."""
    if fmt in ('csv', 'jsonl') and table is None and len(tables) > 1:
        if fmt == 'csv':
            return to_zip(tables, meta, decimal), MEDIA['zip'], 'zip'
        table = next(iter(tables))
    if table is not None:
        if table not in tables:
            raise KeyError(table)
        tables = {table: tables[table]}
    one = next(iter(tables.values()))
    if fmt == 'csv':
        return to_csv(one, decimal), MEDIA['csv'], 'csv'
    if fmt == 'jsonl':
        return to_jsonl(one), MEDIA['jsonl'], 'jsonl'
    if fmt == 'xlsx':
        return to_xlsx(tables, meta), MEDIA['xlsx'], 'xlsx'
    return to_json(tables, meta), MEDIA['json'], 'json'


def meta_for(kind, **fields):
    notes = {'insights': f'{NOT_ADVICE} Слои утверждений: факт, аналитика Банка, вывод AI, гипотеза. Где позиции Банка нет — '
                         f'{NOT_BANK.lower()} Цитаты отчётов Банка сокращены до {QUOTE_CHARS} символов.',
             'payments': PAYMENTS_NOTE, 'counterparties': PAYMENTS_NOTE}
    return dict(kind=kind, generated_at=datetime.now(timezone.utc).isoformat(), note=notes[kind], **fields)
