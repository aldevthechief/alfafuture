"""Locate literal quotes in original articles. Offset/URL metadata is assigned only by code."""
def locate_quote(news: dict, quote: str) -> dict | None:
    if not isinstance(quote, str) or not quote.strip():
        return None
    text = news.get('text') or ''
    start = text.find(quote)
    if start < 0:
        return None
    return dict(source=f"news:{news['id']}", quote=quote, start=start, end=start + len(quote),
                url=news.get('url') or '')


def verify_claim_evidence(insight, news: list[dict], bank: list[dict] | None = None, *, origin: str = 'news') -> list[str]:
    from stage6_insight_generation.insight_writer.insight import SourceExcerpt
    by_ref = {f"news:{row['id']}": row for row in news}
    if origin == 'bank':
        # Bank-only facts quote the real report fragment, never a fabricated news row.
        by_ref = {}
        for row in bank or []:
            ref = row.get('id') or f'bank:{row["report"]}:{row["page"]}'
            by_ref[ref] = dict(row, id=ref)
    issues = []
    for claim in insight.what_happened:
        verified = []
        for excerpt in claim.evidence:
            row = by_ref.get(excerpt.source)
            found = locate_quote(row, excerpt.quote) if row is not None and excerpt.source in claim.sources else None
            if found and origin == 'bank':
                found['source'] = excerpt.source
            if found:
                verified.append(SourceExcerpt(**found))
            else:
                issues.append(f'Недостоверная цитата: {claim.text}')
        claim.evidence = verified
        if not verified:
            issues.append(f'Нет дословного фрагмента {"отчёта" if origin == "bank" else "статьи"}: {claim.text}')
        if any(ref not in by_ref for ref in claim.sources):
            issues.append(f'Неизвестный источник {"тезиса Банка" if origin == "bank" else "новости"}: {claim.text}')
        if not verified or any(ref not in by_ref for ref in claim.sources):
            claim.unconfirmed = True
    return issues
