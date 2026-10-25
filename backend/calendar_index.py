"""Маленький SQLite-индекс дат событий для календаря кабинета.

Обновляется при завершении джоба анализа (см. Service/jobs.on_finish), календарь читает записи
по диапазону дат через GET /v1/calendar. Дата берётся только из структурных полей результата
(insight.event_date, затем дата самой новости); сроки из текста («до конца квартала», «месяцы»)
не превращаются в точную дату — такие строки в индекс не попадают.
Пересборка по сохранённым джобам: python -X utf8 -m backend.calendar_index
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import date, datetime, timezone
from pathlib import Path

from common.data import DATA

DB_PATH = DATA / 'calendar' / 'events.db'
ANALYSES_DIR = DATA / 'api' / 'analyses'
DEMO_DIR = Path(__file__).with_name('demo_cache')

_ISO_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  company_id TEXT NOT NULL,
  role TEXT NOT NULL,
  news_id TEXT NOT NULL,
  event_date TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'calendar',
  title TEXT NOT NULL,
  channel TEXT,
  confidence TEXT,
  impact_level TEXT,
  job_id TEXT,
  user_id TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(company_id, role, news_id)
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(event_date);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _iso(value) -> str | None:
    """Точная дата YYYY-MM-DD или None; всё неоднозначное остаётся без даты."""
    if not isinstance(value, str) or not _ISO_DATE.match(value.strip()):
        return None
    try:
        date.fromisoformat(value.strip())
    except ValueError:
        return None
    return value.strip()


def _row_event(row: dict, request: dict, job_id: str | None) -> dict | None:
    if not isinstance(row, dict) or row.get('status') != 'ready':
        return None
    insight = row.get('insight') or {}
    news_id = row.get('news_id')
    if news_id is None:
        return None
    event_date = _iso(insight.get('event_date'))
    if not event_date:
        news = (row.get('news_references') or {}).get(f'news:{news_id}') or {}
        event_date = _iso(news.get('date'))
    if not event_date:
        return None
    title = str(insight.get('title') or row.get('title') or '').strip()
    if not title:
        return None
    impact = insight.get('impact') or {}
    return dict(company_id=str(request.get('company_id') or ''), role=str(request.get('role') or ''),
                news_id=str(news_id), event_date=event_date, kind='calendar', title=title,
                channel=insight.get('channel') or '', confidence=insight.get('confidence') or '',
                impact_level=impact.get('level') or '', job_id=job_id or '',
                user_id=str(request.get('user_id') or ''), updated_at=_now())


def index_job(job: dict, db_path: Path = DB_PATH) -> int:
    """Записать события одного завершённого джоба анализа; строки компании+роли заменяются."""
    if not isinstance(job, dict) or job.get('kind') not in (None, 'analysis'):
        return 0
    request = job.get('request') or {}
    company_id, role = request.get('company_id'), request.get('role')
    if not company_id or not role:
        return 0
    rows = (job.get('result') or {}).get('results') or []
    events = [e for e in (_row_event(row, request, job.get('id')) for row in rows) if e]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        con = sqlite3.connect(db_path)
        try:
            with con:
                con.executescript(_SCHEMA)
                con.execute('DELETE FROM events WHERE company_id = ? AND role = ?', (company_id, role))
                con.executemany(
                    'INSERT OR REPLACE INTO events(company_id, role, news_id, event_date, kind, title, channel,'
                    ' confidence, impact_level, job_id, user_id, updated_at)'
                    ' VALUES(:company_id, :role, :news_id, :event_date, :kind, :title, :channel, :confidence,'
                    ' :impact_level, :job_id, :user_id, :updated_at)', events)
        finally:
            con.close()
    return len(events)


def maybe_index_job(job: dict, db_path: Path = DB_PATH) -> None:
    """Точка подключения для jobs.on_finish: никогда не роняет пул агентов."""
    try:
        if job.get('kind') == 'analysis' and job.get('status') in ('succeeded', 'partial'):
            index_job(job, db_path=db_path)
    except Exception as exc:  # индекс вторичен; джоб уже сохранён
        print(f'calendar index: skipped job {job.get("id")}: {exc}')


def events_between(date_from: str, date_to: str, company_id: str | None = None,
                   db_path: Path = DB_PATH) -> list[dict]:
    """Записи календаря по диапазону дат [date_from, date_to] включительно, YYYY-MM-DD."""
    if not (_iso(date_from) and _iso(date_to)):
        raise ValueError('from and to must be YYYY-MM-DD')
    if not db_path.exists():
        return []  # ещё ни одного проанализированного события — индекса нет
    where, params = 'event_date >= ? AND event_date <= ?', [date_from, date_to]
    if company_id:
        where += ' AND company_id = ?'
        params.append(company_id)
    with _LOCK:
        con = sqlite3.connect(db_path)
        try:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                f'SELECT company_id, role, news_id, event_date, kind, title, channel, confidence, impact_level, job_id'
                f' FROM events WHERE {where} ORDER BY event_date, company_id, news_id', params).fetchall()
        finally:
            con.close()
    return [dict(r) for r in rows]


def rebuild(db_path: Path = DB_PATH) -> int:
    """Разовый бэкфилл по сохранённым джобам (data/api/analyses) и демо-кешу.

    Только succeeded/partial, по времени создания: у пары компания+роль побеждает свежий прогон."""
    parsed = []
    for path in ([*ANALYSES_DIR.glob('*.json')] if ANALYSES_DIR.is_dir() else []) + \
                ([*DEMO_DIR.glob('*.json')] if DEMO_DIR.is_dir() else []):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if job.get('status') in ('succeeded', 'partial'):
                parsed.append(job)
        except Exception as exc:
            print(f'calendar index: skipped {path.name}: {exc}')
    parsed.sort(key=lambda job: job.get('created_at') or '')
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        con = sqlite3.connect(db_path)
        try:
            with con:
                con.executescript(_SCHEMA)
                con.execute('DELETE FROM events')
        finally:
            con.close()
    return sum(index_job(job, db_path=db_path) for job in parsed)


if __name__ == '__main__':
    print(f'{DB_PATH}: {rebuild()} dated events')
