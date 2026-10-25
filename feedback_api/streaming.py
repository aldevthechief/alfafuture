"""The shared SSE transport: progressive text, durable completion and idempotent concurrent retries."""
import json
import logging
import queue
import threading
from weakref import WeakValueDictionary

from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)


class AnswerStreams:
    def __init__(self):
        self.locks = WeakValueDictionary()
        self.guard = threading.Lock()

    def response(self, key, work):
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        events = queue.Queue()

        def worker():
            try:
                with lock:
                    result = work(lambda text: events.put(('text', {'text':text})))
                    events.put(('complete', result))
            except Exception as exc:
                log.warning('streamed analysis failed: %s', type(exc).__name__)
                status = 503 if str(exc).startswith('set LLM_API_KEY') else 502
                events.put(('error', {'status':status, 'message':'Answer unavailable; the message is saved, retry later'}))
            finally:
                events.put(None)

        def generate():
            threading.Thread(target=worker, daemon=True).start()
            while True:
                try:
                    event = events.get(timeout=15)
                except queue.Empty:
                    yield ': keep-alive\n\n'
                    continue
                if event is None:
                    break
                name, data = event
                yield f'event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

        return StreamingResponse(generate(), media_type='text/event-stream',
                                 headers={'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})
