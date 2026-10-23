"""Bounded, durable job metadata; independent pools keep PDF work away from running agents."""
import contextvars
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common.news_corpus import write_json

class QueueFull(RuntimeError):
    pass

class Jobs:
    def __init__(self, root: Path, workers=4, capacity=64):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.pool=ThreadPoolExecutor(workers, thread_name_prefix=self.root.name)
        self.slots=threading.BoundedSemaphore(capacity)
        self.lock=threading.RLock(); self.items={}; self.futures={}
        for path in self.root.glob('*.json'):
            item=json.loads(path.read_text(encoding='utf-8'))
            if item['status'] in ('queued','running'):
                item.update(status='interrupted',error='Server restarted; submit this job again')
                write_json(path,item)
            self.items[item['id']]=item

    def _save(self,item):
        write_json(self.root/(item['id']+'.json'),item)

    def submit(self,kind,request,work):
        if not self.slots.acquire(blocking=False):
            raise QueueFull('Job queue is full')
        item=dict(id=uuid.uuid4().hex,kind=kind,request=request,status='queued',
                  created_at=datetime.now(timezone.utc).isoformat(),result=None,error=None)
        context=contextvars.copy_context()
        with self.lock:
            self.items[item['id']]=item; self._save(item)
        def run():
            try:
                with self.lock:
                    item['status']='running'; self._save(item)
                result=context.run(work)
                with self.lock:
                    item.update(status='succeeded',result=json.loads(json.dumps(result,default=str)))
            except Exception as exc:
                with self.lock:
                    item.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            finally:
                try:
                    with self.lock:
                        item['finished_at']=datetime.now(timezone.utc).isoformat(); self._save(item)
                finally:
                    self.slots.release()
        try:
            self.futures[item['id']]=self.pool.submit(run)
        except Exception:
            self.slots.release()
            raise
        return self.get(item['id'])

    def get(self,id):
        with self.lock:
            return json.loads(json.dumps(self.items[id]))

    def wait(self,id):
        self.futures[id].result()
        return self.get(id)

    def close(self):
        self.pool.shutdown(wait=True)
