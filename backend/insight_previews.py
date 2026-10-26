"""Small projections of finished results or the checked rows of a running/failed job."""

DONE = ('succeeded', 'partial', 'failed', 'interrupted')


def metadata(job):
    return {key: value for key, value in job.items() if key not in ('result', 'partial_results')}


def rows_of(job):
    result = job.get('result')
    if isinstance(result, dict) and 'results' in result:
        return result['results'] or []
    return job.get('partial_results') or []


def preview_feed(job):
    result = job.get('result') or {}
    cards = []
    for index, row in enumerate(rows_of(job)):
        insight = row.get('insight')
        if row.get('status') != 'ready' or not insight:
            continue
        bank = row.get('source') == 'bank_analytics' or insight.get('origin') == 'bank'
        news = (row.get('news_references') or {}).get(f"news:{row.get('news_id')}", {})
        why = insight.get('why_it_matters') or []
        cards.append(dict(
            id=insight.get('id') or f"row-{index}-{row.get('news_id', 'bank')}",
            job_id=job['id'], result_index=index, preview=True, origin='bank' if bank else 'news',
            title=insight.get('title', ''), lead=why[0].get('text', '') if why else '',
            role=(row.get('evidence') or {}).get('role') or job['request'].get('role') or insight.get('role'),
            published=insight.get('event_date') or news.get('date') or job.get('finished_at'),
            channel=insight.get('channel') or ('Аналитика Банка' if bank else 'Новости'),
            confidence=insight.get('confidence') or 'medium',
            impact_level=(insight.get('impact') or {}).get('level'),
            counts={key: len(insight.get(key) or []) for key in ('risks', 'opportunities', 'actions', 'bank_view')},
        ))
    return dict(computed_at=job.get('finished_at') or (job.get('stage') or {}).get('updated_at') or job['created_at'], insights=cards,
                stage=job.get('stage'), partial=job['status'] not in DONE,
                status=job['status'], errors=result.get('errors') or (
                    [dict(component='job', message=job['error'])] if job.get('error') else []))


def detail(job, index, insight_id=None):
    rows = rows_of(job)
    if insight_id is not None:
        # A merged cache may gain rows between preview and open; keep the selected card's identity.
        index = next((card['result_index'] for card in preview_feed(job)['insights']
                      if card['id'] == insight_id), -1)
    if index < 0 or index >= len(rows) or rows[index].get('status') != 'ready' or not rows[index].get('insight'):
        raise KeyError(index)
    return dict(metadata(job), result=dict(results=[rows[index]]))
