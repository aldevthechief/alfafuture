"""Small feed projections; evidence and full claims are only copied for an opened insight."""


def metadata(job):
    return {key: value for key, value in job.items() if key != 'result'}


def preview_feed(job):
    result = job.get('result') or {}
    cards = []
    for index, row in enumerate(result.get('results', [])):
        insight = row.get('insight')
        if row.get('status') != 'ready' or not insight:
            continue
        bank = row.get('source') == 'bank_analytics' or insight.get('origin') == 'bank'
        news = (row.get('news_references') or {}).get(f"news:{row.get('news_id')}", {})
        why = insight.get('why_it_matters') or []
        cards.append(dict(
            id=insight.get('id') or f"row-{index}-{row.get('news_id', 'bank')}",
            job_id=job['id'], result_index=index, preview=True,
            title=insight.get('title', ''), lead=why[0].get('text', '') if why else '',
            role=(row.get('evidence') or {}).get('role') or job['request'].get('role') or insight.get('role'),
            published=insight.get('event_date') or news.get('date') or job.get('finished_at'),
            channel=insight.get('channel') or ('Аналитика Банка' if bank else 'Новости'),
            confidence=insight.get('confidence') or 'medium',
            impact_level=(insight.get('impact') or {}).get('level'),
            counts={key: len(insight.get(key) or []) for key in ('risks', 'opportunities', 'actions', 'bank_view')},
        ))
    return dict(computed_at=job.get('finished_at') or job['created_at'], insights=cards,
                status=job['status'], errors=result.get('errors') or (
                    [dict(component='job', message=job['error'])] if job.get('error') else []))


def detail(job, index, insight_id=None):
    rows = (job.get('result') or {}).get('results', [])
    if insight_id is not None:
        # A merged cache may gain rows between preview and open; keep the selected card's identity.
        index = next((card['result_index'] for card in preview_feed(job)['insights']
                      if card['id'] == insight_id), -1)
    if index < 0 or index >= len(rows) or rows[index].get('status') != 'ready' or not rows[index].get('insight'):
        raise KeyError(index)
    return dict(metadata(job), result=dict(results=[rows[index]]))
