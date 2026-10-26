"""Bounded, durable job metadata; independent pools keep PDF work away from running agents."""
import contextvars
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from common.news_corpus import write_json
from common import progress

class QueueFull(RuntimeError):
    pass

class Jobs:
    def __init__(self, root: Path, workers=4, capacity=64):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.pool=ThreadPoolExecutor(workers, thread_name_prefix=self.root.name)
        self.slots=threading.BoundedSemaphore(capacity)
        self.lock=threading.RLock(); self.changed=threading.Condition(self.lock)
        self.items={}; self.futures={}; self.versions={}
        self.on_finish=None  # optional callback(item) after a job reaches a final state; never raises
        for path in self.root.glob('*.json'):
            item=json.loads(path.read_text(encoding='utf-8'))
            if item['status'] in ('queued','running'):
                item.update(status='interrupted',error='Server restarted; submit this job again')
                if item.get('stage'):
                    item['stage'].update(code='interrupted',label=progress.TERMINAL['interrupted'])
                write_json(path,item)
            self.items[item['id']]=item
            self.versions[item['id']]=item.get('revision',1)

    def _save(self,item):
        item['revision']=item.get('revision',0)+1
        write_json(self.root/(item['id']+'.json'),item)
        self.versions[item['id']]=item['revision']
        self.changed.notify_all()

    def report(self,id,partial_results):
        with self.lock:
            item=self.items[id]
            if item['status']!='running': return
            item['partial_results']=json.loads(json.dumps(partial_results,default=str))
            self._save(item)

    def wait_change(self,id,version,timeout,project=None):
        with self.changed:
            if id not in self.items: raise KeyError(id)
            self.changed.wait_for(lambda:self.versions.get(id,0)!=version,timeout)
            return self.versions.get(id,0),self.get(id,project)

    def submit(self,kind,request,work,start=None,plan=None):
        if not self.slots.acquire(blocking=False):
            raise QueueFull('Job queue is full')
        item=dict(id=uuid.uuid4().hex,kind=kind,request=request,status='queued',
                  created_at=datetime.now(timezone.utc).isoformat(),result=None,error=None)
        def publish(snapshot):
            with self.lock:
                if snapshot['seq'] >= (item.get('stage') or {}).get('seq',-1):
                    item['stage']=snapshot; self._save(item)
        stage=progress.Progress(publish,plan=[start] if start else plan,
                                on_results=lambda rows:self.report(item['id'],rows))
        item['stage']=stage.snapshot()
        context=contextvars.copy_context()
        with self.lock:
            self.items[item['id']]=item; self._save(item)
        def bound():
            progress.bind(stage)
            if start: stage.enter(start)
            return work()
        def run():
            try:
                with self.lock:
                    item['status']='running'; self._save(item)
                result=context.run(bound)
                with self.lock:
                    partial=isinstance(result,dict) and result.get('status')=='partial'
                    error='; '.join(e['message'] for e in result.get('errors',[])) if partial else None
                    item.update(status='partial' if partial else 'succeeded',error=error,
                                result=json.loads(json.dumps(result,default=str)))
                    item.pop('partial_results',None)
                stage.finish('done')
            except Exception as exc:
                with self.lock:
                    item.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                stage.finish('failed',error=item['error'])
            finally:
                try:
                    with self.lock:
                        item['finished_at']=datetime.now(timezone.utc).isoformat(); self._save(item)
                finally:
                    if self.on_finish is not None:
                        try: self.on_finish(item)
                        except Exception as exc: print(f'{self.root.name}: on_finish failed for {item["id"]}: {exc}')
                    self.slots.release()
        try:
            self.futures[item['id']]=self.pool.submit(run)
        except Exception:
            self.slots.release()
            raise
        return self.get(item['id'])

    def get(self,id,project=None):
        with self.lock:
            item=self.items[id]
            return json.loads(json.dumps(project(item) if project is not None else item))

    def wait(self,id):
        self.futures[id].result()
        return self.get(id)

    def close(self):
        self.pool.shutdown(wait=True)
