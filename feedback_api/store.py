"""SQLite storage of registered insights and feedback; short transactions, LLM calls never hold one.
Ported from the maxon_dialogue branch (aldev); versions / revisions are not ported yet."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from common.personalization import Personalization, ProfileAnswer, Reaction, UserNote
from feedback_api.schemas import Analysis, FeedbackInput, FeedbackRecord, InsightRecord, Kind, ReactionSummary, RegisterInsight


# one verdict per insight: the latest of these wins, so 👍 changed to «Уже знали» withdraws the like (and back)
RATINGS = {Kind.like, Kind.useful, Kind.dislike, Kind.not_relevant, Kind.already_known}


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reaction_of(kind: str, record: InsightRecord) -> Reaction:
    """What a reaction is about: the channel and title give the topic; title + what happened give the similarity text
    (why_it_matters is left out: it repeats company facts that every insight of the company shares)."""
    i = record.insight
    return Reaction(kind=kind, news_id=record.news_id, channel=i.channel, title=i.title,
                    text=' '.join([i.title, *(c.text for c in i.what_happened[:2])]))


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys = ON')
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('PRAGMA journal_mode = WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, company_id TEXT NOT NULL,
                    role TEXT NOT NULL, news_id INTEGER, document TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, insight_id TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL, comment TEXT NOT NULL, emoji TEXT NOT NULL, scope TEXT NOT NULL,
                    request_id TEXT NOT NULL, analysis TEXT, created_at TEXT NOT NULL,
                    selected_text TEXT NOT NULL DEFAULT '', selected_source TEXT NOT NULL DEFAULT 'insight',
                    UNIQUE(insight_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS insight_owner ON insights(user_id, company_id, role);
                CREATE INDEX IF NOT EXISTS feedback_history ON feedback(insight_id, created_at);
                CREATE TABLE IF NOT EXISTS profile_answers (
                    user_id TEXT NOT NULL, company_id TEXT NOT NULL, question_id TEXT NOT NULL,
                    statement TEXT NOT NULL, verdict TEXT NOT NULL, text TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, company_id, question_id)
                );
            ''')
            db.execute('BEGIN IMMEDIATE')  # serialize the migration if several workers start together
            if 'selected_text' not in {row['name'] for row in db.execute('PRAGMA table_info(feedback)')}:
                db.execute("ALTER TABLE feedback ADD COLUMN selected_text TEXT NOT NULL DEFAULT ''")
            if 'selected_source' not in {row['name'] for row in db.execute('PRAGMA table_info(feedback)')}:
                db.execute("ALTER TABLE feedback ADD COLUMN selected_source TEXT NOT NULL DEFAULT 'insight'")

    # A user's company facts hold across reader roles; bank facts are never overwritten.
    def set_answer(self, user_id: str, company_id: str, answer: ProfileAnswer) -> ProfileAnswer:
        with self.connection() as db:
            db.execute('''INSERT INTO profile_answers VALUES (?, ?, ?, ?, ?, ?, ?)
                          ON CONFLICT(user_id, company_id, question_id) DO UPDATE SET
                          statement=excluded.statement, verdict=excluded.verdict, text=excluded.text, updated_at=excluded.updated_at''',
                       (user_id, company_id, answer.question_id, answer.statement, answer.verdict, answer.text, now()))
        return answer

    def delete_answer(self, user_id: str, company_id: str, question_id: str):
        with self.connection() as db:
            db.execute('DELETE FROM profile_answers WHERE user_id=? AND company_id=? AND question_id=?',
                       (user_id, company_id, question_id))

    def answers(self, user_id: str, company_id: str) -> list[ProfileAnswer]:
        with self.connection() as db:
            # The pipeline can read an old database before feedback-api has run its migration.
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='profile_answers'").fetchone():
                return []
            rows = db.execute('''SELECT question_id, statement, verdict, text FROM profile_answers
                                 WHERE user_id=? AND company_id=? ORDER BY updated_at DESC LIMIT 30''', (user_id, company_id)).fetchall()
        return [ProfileAnswer(**dict(row)) for row in rows]

    # --- insights ---
    def register(self, user_id: str, body: RegisterInsight) -> InsightRecord:
        news_id = body.news_id
        if news_id is None and body.insight.trigger.news_ids:
            news_id = body.insight.trigger.news_ids[0]
        key = json.dumps([user_id, body.company_id, body.insight.role, body.registration_key], ensure_ascii=False)
        identifier = str(uuid5(NAMESPACE_URL, key)) if body.registration_key else str(uuid4())
        record = InsightRecord(id=identifier, user_id=user_id, company_id=body.company_id, news_id=news_id,
                               insight=body.insight, evidence=body.evidence,
                               card_sources={} if body.evidence else body.card_sources, created_at=now())
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO insights VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (record.id, user_id, record.company_id, record.insight.role, record.news_id,
                        record.model_dump_json(), record.created_at))
            existing = InsightRecord.model_validate_json(db.execute(
                'SELECT document FROM insights WHERE id=?', (record.id,)).fetchone()['document'])
            if (existing.user_id != user_id or existing.news_id != news_id or existing.insight != body.insight):
                raise Conflict('registration_key already belongs to a different insight')
        return existing

    def insight(self, user_id: str, insight_id: str) -> InsightRecord:
        with self.connection() as db:
            row = db.execute('SELECT document FROM insights WHERE id=? AND user_id=?', (insight_id, user_id)).fetchone()
        if row is None:
            raise NotFound()
        return InsightRecord.model_validate_json(row['document'])

    def insights(self, user_id: str, company_id: str, role: str, limit: int = 50) -> list[InsightRecord]:
        with self.connection() as db:
            rows = db.execute('''SELECT document FROM insights WHERE user_id=? AND company_id=? AND role=?
                                 ORDER BY created_at DESC LIMIT ?''', (user_id, company_id, role, limit)).fetchall()
        return [InsightRecord.model_validate_json(r['document']) for r in rows]

    # --- feedback ---
    @staticmethod
    def feedback_record(row) -> FeedbackRecord:
        data = dict(row)
        data.pop('news_id', None)
        data['analysis'] = json.loads(data['analysis']) if data['analysis'] else None
        return FeedbackRecord.model_validate(data)

    def add(self, user_id: str, insight_id: str, body: FeedbackInput) -> FeedbackRecord:
        self.insight(user_id, insight_id)  # ownership check
        with self.connection() as db:
            db.execute('''INSERT OR IGNORE INTO feedback
                          (id, insight_id, kind, comment, emoji, scope, request_id, analysis, created_at, selected_text, selected_source)
                          VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)''',
                       (str(uuid4()), insight_id, body.kind, body.comment, body.emoji, body.scope, body.request_id, now(), body.selected_text, body.selected_source))
            row = db.execute('SELECT * FROM feedback WHERE insight_id=? AND request_id=?', (insight_id, body.request_id)).fetchone()
            record = self.feedback_record(row)
            if (record.kind, record.comment, record.emoji, record.scope, record.selected_text, record.selected_source) != (body.kind, body.comment, body.emoji, body.scope, body.selected_text, body.selected_source):
                raise Conflict('request_id already used with a different payload')
        return record

    def feedback(self, user_id: str, feedback_id: str) -> FeedbackRecord:
        with self.connection() as db:
            row = db.execute('''SELECT f.* FROM feedback f JOIN insights i ON i.id=f.insight_id
                                WHERE f.id=? AND i.user_id=?''', (feedback_id, user_id)).fetchone()
        if row is None:
            raise NotFound()
        return self.feedback_record(row)

    def history(self, user_id: str, insight_id: str, limit: int = 50, offset: int = 0) -> list[FeedbackRecord]:
        self.insight(user_id, insight_id)
        with self.connection() as db:
            rows = db.execute('SELECT * FROM feedback WHERE insight_id=? ORDER BY created_at, id LIMIT ? OFFSET ?',
                              (insight_id, limit, offset)).fetchall()
        return [self.feedback_record(r) for r in rows]

    def recent(self, user_id: str, feedback: FeedbackRecord, n: int = 8) -> list[FeedbackRecord]:
        """The `n` messages before `feedback` on its insight, oldest first: the context of a follow-up question."""
        self.insight(user_id, feedback.insight_id)
        with self.connection() as db:
            rows = db.execute('''SELECT * FROM feedback WHERE insight_id=? AND id<>? AND (created_at, id) < (?, ?)
                                 ORDER BY created_at DESC, id DESC LIMIT ?''',
                              (feedback.insight_id, feedback.id, feedback.created_at, feedback.id, n)).fetchall()
        return [self.feedback_record(r) for r in reversed(rows)]

    def save_analysis(self, user_id: str, feedback_id: str, analysis: Analysis) -> FeedbackRecord:
        self.feedback(user_id, feedback_id)
        with self.connection() as db:  # concurrent identical processing keeps the first successful interpretation
            db.execute('UPDATE feedback SET analysis=? WHERE id=? AND analysis IS NULL', (analysis.model_dump_json(), feedback_id))
        return self.feedback(user_id, feedback_id)

    def delete(self, user_id: str, feedback_id: str):
        self.feedback(user_id, feedback_id)
        with self.connection() as db:
            db.execute('DELETE FROM feedback WHERE id=?', (feedback_id,))

    def reactions(self, user_id: str, insight_id: str) -> ReactionSummary:
        rows = self.history(user_id, insight_id, 500)
        counts = {}
        for f in rows:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return ReactionSummary(insight_id=insight_id, counts=counts, mine=sorted({f.kind for f in rows}))

    # --- the signal for the next selection ---
    def personalization(self, user_id: str, company_id: str, role: str, limit: int = 300) -> Personalization:
        with self.connection() as db:
            rows = db.execute('''SELECT f.*, i.document FROM feedback f JOIN insights i ON i.id=f.insight_id
                                 WHERE i.user_id=? AND i.company_id=? AND i.role=? AND f.scope='future'
                                 ORDER BY f.created_at DESC, f.id LIMIT ?''', (user_id, company_id, role, limit)).fetchall()
        reactions, notes, rated = [], [], set()
        for row in rows:  # newest first
            data = dict(row)
            record = InsightRecord.model_validate_json(data.pop('document'))
            f = self.feedback_record(data)
            if f.kind in RATINGS:  # repeated clicks do not add up either
                if record.id in rated:
                    continue
                rated.add(record.id)
            reactions.append(reaction_of(f.kind, record))
            if f.analysis and f.analysis.context_notes:
                notes += [UserNote(source_id=f'user:{f.id}', kind=n.kind, text=n.text, polarity=n.polarity)
                          for n in f.analysis.context_notes]
            elif f.comment and f.kind not in (Kind.question,):
                notes.append(UserNote(source_id=f'user:{f.id}', kind=f.kind, text=f.comment))
        pers = Personalization.from_reactions(user_id, company_id, role, list(reversed(reactions)), notes[:40])
        pers.answers = self.answers(user_id, company_id)
        return pers
