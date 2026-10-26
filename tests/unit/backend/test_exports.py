"""Exports of insights, payments and counterparties: synthetic company tables and result rows, no model calls."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.service import Service

def load_workbook(stream):
    """Minimal XLSX reader for the tests (no extra dependency): sheet name -> rows of values; `.types` keeps cell types."""
    import re
    import xml.etree.ElementTree as ET
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    z = zipfile.ZipFile(stream)
    strings = [''.join(t.text or '' for t in si.iter('{%s}t' % ns['m']))
               for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
    names = [s.get('name') for s in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets', ns)]
    book = Book(); book.sheetnames = names
    for i, name in enumerate(names, 1):
        rows, types = [], {}
        for row in ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml')).iter('{%s}row' % ns['m']):
            values = {}
            for c in row.findall('m:c', ns):
                col = re.match(r'[A-Z]+', c.get('r')).group()
                index = 0
                for ch in col: index = index * 26 + ord(ch) - 64
                v, f = c.find('m:v', ns), c.find('m:f', ns)
                types[c.get('r')] = 'f' if f is not None else c.get('t', 'n')
                values[index - 1] = strings[int(v.text)] if c.get('t') == 's' else (float(v.text) if v is not None else None)
            rows.append(tuple(values.get(j) for j in range(max(values, default=-1) + 1)))
        book[name] = Sheet(rows, types)
    return book


class Book(dict):
    sheetnames = []


class Sheet:
    def __init__(self, rows, types):
        self.values, self.types = rows, types


ROW = dict(status='ready', news_id=7, source='news', news_references={'news:7': dict(title='ЦБ снизил ставку', date='2026-10-24',
                                                                                     url='https://example.org/7')},
           insight=dict(id='card-7', title='=HYPERLINK("http://evil")', role='cfo', company_id='U_1', event_date='2026-10-24',
                        what_happened=[dict(id='e1', text='ЦБ снизил ставку до 15%', kind='fact', sources=['news:7'])],
                        why_it_matters=[dict(id='w1', text='Долг подешевеет', kind='ai', sources=['fact:1'], depends_on=['h1'])],
                        bank_view=[dict(id='b1', text='Банк ждёт 14% к концу года', kind='bank', sources=['bank:FI.pdf:p3.1'],
                                        evidence=[dict(source='bank:FI.pdf:p3.1', quote='Ожидаем ' + 'очень ' * 200 + 'снижение',
                                                       title='FI Strategy', date='2026-07-22', page=3, status='exact')])],
                        hypotheses=[dict(id='h1', text='Ставка по кредиту плавающая', kind='hypothesis', sources=[], check='договор')],
                        risks=[], opportunities=[], actions=['Пересмотреть график'], impact=dict(level='high', criteria='x', estimate='~12 млн ₽'),
                        confidence='medium', disclaimer='Не является инвестиционной рекомендацией.'))
BANK_ROW = dict(status='ready', source='bank_analytics', insight=dict(id='card-b', title='Рубль', role='cfo', origin='bank',
                                                                      event_date='2026-09-01', bank_view=[]))


def tables():
    U = pd.DataFrame(dict(SYNTHETIC_U=['U_1', 'U_2', 'U_3', 'U_4'], GROUP_ID=['g1', 'g1', 'g2', 'g3'], okved=['46.51', '41.20', '10.11', '47.11'],
                          MAINOKVED_NAME=['Опт', 'Стройка', 'Мясо', 'Розница'], REGION=['Москва', 'Москва', 'Омск', 'Тула'],
                          CL_TYPE=['CLIENT', 'CLIENT', 'PROSPECT', 'CLIENT'], REVENUE_COMPANY=[2e10, 5e9, 1e9, 1e8],
                          AMOUNT_WORKERS_FNS_GRADATION=['Более 251', '101-250', '16-100', 'Менее 15'])).set_index('SYNTHETIC_U', drop=False)
    P = pd.DataFrame(dict(PAYER=['U_1', 'U_1', 'U_3', 'U_1'], BENEFICIARY=['U_2', 'U_3', 'U_1', 'U_X'],
                          PAY_AMT=[600., 300., 50., 100.], PAY_CNT=[6, 3, 1, 2]))
    S = pd.DataFrame(dict(PAYER=['U_1'], BENEFICIARY=['U_1'], PAY_AMT=[9000.], PAY_CNT=[30])).set_index('PAYER')
    F = pd.DataFrame(dict(SYNTHETIC_F=['F_1', 'F_1'], SYNTHETIC_U=['U_1', 'U_4']))
    return U, P, S, F


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = Service(Path(self.tmp.name))
        self.service.case_tables = tables
        self.service.agents.items['a' * 32] = dict(id='a' * 32, kind='analysis', status='succeeded', created_at='2026-10-25',
            finished_at='2026-10-25T10:00', error=None, request=dict(company_id='U_1', role='cfo', user_id='alice'),
            result=dict(results=[ROW, BANK_ROW, dict(status='reject')], bank_version='b' * 64, corpus_fingerprint='c' * 64))
        self.client = TestClient(create_app(self.service))
        self.client.__enter__()
        self.alice = {'X-User-ID': 'alice'}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def get(self, path, **params):
        response = self.client.get(path, params=params, headers=self.alice)
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def test_insight_xlsx_keeps_layers_short_bank_quotes_disclaimers_and_no_formulas(self):
        response = self.get('/v1/exports/insights', job_id='a' * 32)
        self.assertIn('attachment; filename="alfa-insights_U_1_cfo_', response.headers['content-disposition'])
        book = load_workbook(io.BytesIO(response.content))
        self.assertEqual(book.sheetnames, ['Инсайты', 'Утверждения', 'Источники', 'Действия', 'О выгрузке'])
        insights = list(book['Инсайты'].values)
        self.assertEqual(len(insights), 3)  # header + 2 ready insights; the rejected row is not exported
        self.assertEqual(insights[1][3], '=HYPERLINK("http://evil")')
        self.assertEqual(book['Инсайты'].types['D2'], 's')  # a string cell, not a formula
        layers = {row[3] for row in list(book['Утверждения'].values)[1:]}
        self.assertEqual(layers, {'Факт', 'Вывод AI', 'Аналитика Банка', 'Гипотеза'})
        quote = [r for r in list(book['Источники'].values)[1:] if r[3] == 'Отчёт Банка'][0][8]
        self.assertLessEqual(len(quote), 300)
        bank_card = [r for r in insights[1:] if r[0] == 'card-b'][0]
        self.assertIn('Не является официальной позицией Банка', bank_card[16])
        about = dict(list(book['О выгрузке'].values))
        self.assertEqual(about['bank_version'], 'b' * 64)

    def test_csv_is_excel_ready_and_escapes_formulas(self):
        data = self.get('/v1/exports/insights', company_id='U_1', role='cfo', format='csv', table='insights').content
        self.assertTrue(data.startswith('﻿'.encode()))
        rows = list(csv.reader(io.StringIO(data.decode('utf-8-sig')), delimiter=';'))
        self.assertEqual(rows[0][0], 'ID инсайта')
        self.assertEqual(rows[1][3], '\'=HYPERLINK("http://evil")')
        bundle = zipfile.ZipFile(io.BytesIO(self.get('/v1/exports/insights', job_id='a' * 32, format='csv').content))
        self.assertEqual(sorted(bundle.namelist()), ['actions.csv', 'claims.csv', 'insights.csv', 'meta.json', 'sources.csv'])

    def test_json_filters_by_origin_and_date(self):
        body = self.get('/v1/exports/insights', job_id='a' * 32, format='json', origin='bank').json()
        self.assertEqual([r['insight_id'] for r in body['tables']['insights']], ['card-b'])
        body = self.get('/v1/exports/insights', job_id='a' * 32, format='json', date_from='2026-10-01').json()
        self.assertEqual([r['insight_id'] for r in body['tables']['insights']], ['card-7'])
        self.assertIn('инвестиционной', body['meta']['note'])

    def test_payments_keep_directions_self_transfers_and_unknown_counterparties(self):
        rows = self.get('/v1/exports/payments', company_id='U_1', format='json').json()['tables']['payments']
        by = {(r['direction'], r['counterparty_id']): r for r in rows}
        self.assertEqual(by[('Исходящие', 'U_2')]['share'], 0.6)
        self.assertTrue(by[('Исходящие', 'U_2')]['same_group'])
        self.assertFalse(by[('Исходящие', 'U_X')]['in_table'])
        self.assertEqual(by[('Входящие', 'U_3')]['amount'], 50.0)
        self.assertEqual(by[('Между своими счетами', 'U_1')]['amount'], 9000.0)
        group = self.get('/v1/exports/payments', company_id='U_1', scope='group', side='in', format='json').json()
        self.assertEqual({r['company_id'] for r in group['tables']['payments']}, {'U_1', 'U_2'})
        csv_rows = self.get('/v1/exports/payments', company_id='U_1', format='csv').content.decode('utf-8-sig')
        self.assertIn(';0,6;', csv_rows)

    def test_counterparties_concentration_and_group(self):
        book = load_workbook(io.BytesIO(self.get('/v1/exports/counterparties', company_id='U_1').content))
        self.assertEqual(book.sheetnames, ['Контрагенты', 'Концентрация', 'Группа', 'О выгрузке'])
        conc = list(book['Концентрация'].values)[1:]
        suppliers = [r for r in conc if r[1].startswith('Поставщики')][0]
        self.assertEqual(suppliers[3], 3.0)
        self.assertEqual(list(book['Группа'].values)[1][1], 'g1')

    def test_access_file_limits_clients_and_staff_sees_everything(self):
        (Path(self.tmp.name) / 'access.json').write_text(json.dumps({'alice': ['g1']}))
        with TestClient(create_app(self.service, staff_token='secret')) as client:
            self.assertEqual(client.get('/v1/exports/payments', params=dict(company_id='U_2', format='json'), headers=self.alice).status_code, 200)
            self.assertEqual(client.get('/v1/exports/payments', params=dict(company_id='U_3'), headers=self.alice).status_code, 404)
            staff = dict(self.alice, Authorization='Bearer secret')
            self.assertEqual(client.get('/v1/exports/payments', params=dict(company_id='U_3'), headers=staff).status_code, 200)
        self.assertEqual(self.client.get('/v1/exports/insights', params=dict(job_id='a' * 32), headers={'X-User-ID': 'bob'}).status_code, 404)
        self.assertEqual(self.client.get('/v1/exports/payments', params=dict(company_id='U_404')).status_code, 404)
        self.assertEqual(self.client.get('/v1/exports/insights').status_code, 422)

    def test_batch_export_runs_in_background_and_merges_rows(self):
        batch = dict(items=[dict(kind='payments', company_id='U_1'), dict(kind='payments', company_id='U_2'),
                            dict(kind='insights', job_id='a' * 32)], format='xlsx')
        job = self.client.post('/v1/exports', json=batch, headers=self.alice).json()
        done = self.service.exports.wait(job['id'])
        self.assertEqual(done['status'], 'succeeded', done)
        self.assertEqual(done['result']['rows']['insights'], 2)
        self.assertEqual(self.client.get(f'/v1/exports/{job["id"]}/file', headers={'X-User-ID': 'bob'}).status_code, 404)
        book = load_workbook(io.BytesIO(self.client.get(f'/v1/exports/{job["id"]}/file', headers=self.alice).content))
        self.assertEqual({r[0] for r in list(book['Платежи'].values)[1:]}, {'U_1', 'U_2'})


if __name__ == '__main__':
    unittest.main()
