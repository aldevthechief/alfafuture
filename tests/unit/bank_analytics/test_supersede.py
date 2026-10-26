"""A newer own-view forecast of the bank supersedes an older one on the same (indicator, year, unit); no data, no models."""
from stage5_insight_detection.bank_analytics.supersede import mark_chunks, superseded


def frag(fid, date, value, own=True, period='конец 2026', indicator='ставка', kind='forecast', verified=True):
    return {'id': fid, 'report': f'Отчёт {date}', 'date': date, 'own_view': own,
            'figures': [{'value': value, 'unit': '%', 'indicator': indicator, 'kind': kind, 'period': period, 'verified': verified}]}


def test_newer_report_supersedes_different_value():
    rows = [frag('jul', '2026-07-22', 13.5), frag('sep', '2026-09-11', 14.0)]
    assert superseded(rows) == {'jul': 'Отчёт 2026-09-11 (2026-09-11)'}


def test_same_value_other_year_actuals_and_quotes_are_kept():
    rows = [frag('jul', '2026-07-22', 14.0), frag('sep', '2026-09-11', 14.0),     # agrees with the newer view
            frag('jul27', '2026-07-22', 11.0, period='2027'),                    # no newer forecast for 2027
            frag('fact', '2026-07-22', 18.0, kind='actual'),                       # statistics, not a view
            frag('cb', '2026-07-22', 13.0, own=False),                             # a quote of the CB, not the bank
            frag('chart', '2026-07-22', 12.0, verified=False)]                     # unverified chart read
    assert superseded(rows) == {}


def test_chunks_with_a_superseded_fragment_become_stale():
    rows = [frag('jul', '2026-07-22', 13.5), frag('sep', '2026-09-11', 14.0)]
    chunks = [{'id': 'c1', 'frag_ids': ['jul', 'x'], 'stale': False}, {'id': 'c2', 'frag_ids': ['sep'], 'stale': False}]
    assert mark_chunks(chunks, rows) == 1
    assert chunks[0]['stale'] and chunks[0]['superseded_by'].startswith('Отчёт 2026-09-11') and not chunks[1]['stale']


def test_different_horizons_measures_and_currencies_do_not_replace_each_other():
    old=frag('old','2026-07-22',13.5)
    for period in ('среднегодовая 2026','1 квартал 2026','2027','2026–2027'):
        assert superseded([old,frag('new','2026-09-11',14,period=period)])=={}
    change=frag('new','2026-09-11',1)
    change['figures'][0]['measure']='change'
    assert superseded([old,change])=={}
    usd=frag('usd','2026-07-22',90,indicator='курс')
    usd['figures'][0].update(unit='₽',sentence='Прогноз курса доллара')
    eur=frag('eur','2026-09-11',100,indicator='курс')
    eur['figures'][0].update(unit='₽',sentence='Прогноз курса евро')
    assert superseded([usd,eur])=={}
    average=frag('avg','2026-09-11',14,period='2026')
    average['figures'][0]['sentence']='Среднегодовая ключевая ставка'
    assert superseded([old,average])=={}


def test_ambiguous_or_unchecked_new_values_do_not_replace_a_forecast():
    old=frag('old','2026-07-22',13.5)
    for value in (float('nan'),float('inf'),None,'14',True):
        assert superseded([old,frag('new','2026-09-11',value)])=={}
    unverified=frag('new','2026-09-11',14)
    unverified['figures'][0]['status']='approx'
    assert superseded([old,unverified])=={}
    conflicting=[frag('a','2026-09-11',14),frag('b','2026-09-11',15)]
    assert superseded([old,*conflicting])=={}
    invalid=frag('new','not-a-date',14)
    assert superseded([old,invalid])=={}


def test_published_views_and_library_do_not_mutate_older_snapshots():
    from types import SimpleNamespace
    import numpy as np
    from backend.catalog import BankView
    from stage5_insight_detection.bank_analytics.library import BankLibrary,Fragment
    def part(row):
        chunk=dict(id=row['id']+'+c1',frag_ids=[row['id']],report=row['report'],date=row['date'],
                   text='Мы ожидаем ключевую ставку',own_view=True,stale=False,freshness='fresh',
                   kind='text',page=1,pages=[1],seg=None,start=0,end=30)
        return SimpleNamespace(rows=[row],chunks=[chunk],available=[row['report']],twins={},D=np.array([[1.,0.]]))
    old,new=part(frag('old','2026-07-22',13.5)),part(frag('new','2026-09-11',14))
    before=BankView([old]);after=BankView([old,new])
    assert not before.chunks[0]['stale'] and not old.chunks[0]['stale']
    assert after.chunks[0]['stale'] and after.rows[0]['superseded_by']
    assert after._context_chunks[0]['stale']
    assert 'superseded_by' not in old.rows[0]
    fragments=[Fragment(**dict(r,file=r['id']+'.pdf',page=1,n=1,heading='',text='Мы ожидаем ставку',
                               topics=['ставка'],freshness='fresh',age_days=10)) for r in (old.rows[0],new.rows[0])]
    first=BankLibrary(fragments[:1]);combined=BankLibrary(fragments)
    assert not first.fragments[0].stale and not fragments[0].stale
    assert combined.fragments[0].stale and combined.fragments[0].superseded_by
    assert not BankLibrary([combined.fragments[0]]).fragments[0].stale


def test_calculator_rejects_superseded_forecast_and_tools_show_its_replacement():
    from types import SimpleNamespace
    from stage5_insight_detection.insight_judge.calculations import JudgeCalculations
    from stage5_insight_detection.analyst_loop.compute import checked_bank
    from stage5_insight_detection.bank_analytics.tools import BankTools
    old,new=frag('old','2026-07-22',13.5),frag('new','2026-09-11',14)
    for value in (old,new): value.update(kind='text',text='Мы ожидаем ключевую ставку',page=1)
    chunks=[dict(id=value['id']+'+c1',frag_ids=[value['id']],stale=False,freshness='fresh',own_view=True,
                 text=value['text'],report=value['report'],date=value['date'],page=1,pages=[1]) for value in (old,new)]
    mark_chunks(chunks,[old,new])
    bank=SimpleNamespace(rows=[old,new],chunks=chunks,index={c['id']:n for n,c in enumerate(chunks)},twins={},
        passages=lambda *args,**kwargs:[dict(report=old['report'],date=old['date'],pages=[1],stale=True,freshness='stale',
            ids=[chunks[0]['id']],text=old['text'],superseded_by=old['superseded_by'])])
    calc=JudgeCalculations({},'U_1',bank)
    calc.observe_bank([dict(ids=[c['id'] for c in chunks])])
    dossier=SimpleNamespace(bank=calc._bank(),event=SimpleNamespace(date=None))
    assert [fid for fid,_ in checked_bank(dossier)]==['new']
    assert 'прогноз заменён' in BankTools(bank).search_bank_analytics('ставка')
