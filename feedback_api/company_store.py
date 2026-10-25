"""Separate tables for company chats; no rows enter insight feedback or personalization."""
import json
from uuid import uuid4

from feedback_api.schemas import CompanyChatMessage, CompanyConversation
from feedback_api.store import Conflict, NotFound, now


class CompanyChats:
    def __init__(self, store):
        self.store = store

    def initialize(self):
        with self.store.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS company_chats (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, company_id TEXT NOT NULL,
                    role TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(user_id, company_id, role)
                );
                CREATE TABLE IF NOT EXISTS company_chat_messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES company_chats(id),
                    request_id TEXT NOT NULL, document TEXT NOT NULL, analysis TEXT, created_at TEXT NOT NULL,
                    UNIQUE(conversation_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS company_chat_history
                    ON company_chat_messages(conversation_id, created_at, id);
            ''')

    def open(self, user_id, company_id, role):
        with self.store.connection() as db:
            db.execute('INSERT OR IGNORE INTO company_chats VALUES (?, ?, ?, ?, ?)',
                       (str(uuid4()), user_id, company_id, role, now()))
            row = db.execute('SELECT * FROM company_chats WHERE user_id=? AND company_id=? AND role=?',
                             (user_id, company_id, role)).fetchone()
        return self._conversation(row)

    @staticmethod
    def _conversation(row):
        if row is None:
            raise NotFound()
        data = dict(row)
        data.pop('user_id')
        return CompanyConversation.model_validate(data)

    def conversation(self, user_id, conversation_id):
        with self.store.connection() as db:
            row = db.execute('SELECT * FROM company_chats WHERE id=? AND user_id=?', (conversation_id, user_id)).fetchone()
        return self._conversation(row)

    @staticmethod
    def _message(row):
        if row is None:
            raise NotFound()
        data = json.loads(row['document'])
        data['analysis'] = json.loads(row['analysis']) if row['analysis'] else None
        return CompanyChatMessage.model_validate(data)

    def message(self, user_id, message_id):
        with self.store.connection() as db:
            row = db.execute('''SELECT m.* FROM company_chat_messages m JOIN company_chats c ON c.id=m.conversation_id
                                WHERE m.id=? AND c.user_id=?''', (message_id, user_id)).fetchone()
        return self._message(row)

    def add(self, user_id, conversation_id, body, context_factory):
        self.conversation(user_id, conversation_id)

        def check(row):
            message = self._message(row)
            if (message.comment, message.selected_text) != (body.question, body.selected_text):
                raise Conflict('request_id already used with a different payload')
            return message

        with self.store.connection() as db:
            row = db.execute('SELECT * FROM company_chat_messages WHERE conversation_id=? AND request_id=?',
                             (conversation_id, body.request_id)).fetchone()
        if row is not None:
            return check(row)  # exact retry retains its original context, even if company data has changed
        message = CompanyChatMessage(id=str(uuid4()), conversation_id=conversation_id, comment=body.question,
                                     selected_text=body.selected_text, request_id=body.request_id,
                                     context=context_factory(), created_at=now())
        with self.store.connection() as db:
            db.execute('INSERT OR IGNORE INTO company_chat_messages VALUES (?, ?, ?, ?, NULL, ?)',
                       (message.id, conversation_id, body.request_id, message.model_dump_json(exclude={'analysis'}), message.created_at))
            row = db.execute('SELECT * FROM company_chat_messages WHERE conversation_id=? AND request_id=?',
                             (conversation_id, body.request_id)).fetchone()
        return check(row)

    def history(self, user_id, conversation_id, limit=50, before=None):
        self.conversation(user_id, conversation_id)
        where, args = '', [conversation_id]
        if before:
            cursor = self.message(user_id, before)
            if cursor.conversation_id != conversation_id:
                raise NotFound()
            where = ' AND (created_at, id) < (?, ?)'
            args += [cursor.created_at, cursor.id]
        with self.store.connection() as db:
            rows = db.execute('SELECT * FROM company_chat_messages WHERE conversation_id=?' + where +
                              ' ORDER BY created_at DESC, id DESC LIMIT ?', [*args, limit + 1]).fetchall()
        return [self._message(row) for row in reversed(rows[:limit])], len(rows) > limit

    def recent(self, user_id, message):
        return self.history(user_id, message.conversation_id, limit=8, before=message.id)[0]

    def save_analysis(self, user_id, message_id, analysis):
        self.message(user_id, message_id)
        with self.store.connection() as db:
            db.execute('UPDATE company_chat_messages SET analysis=? WHERE id=? AND analysis IS NULL',
                       (analysis.model_dump_json(), message_id))
        return self.message(user_id, message_id)
