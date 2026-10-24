"""Each PDF gets an immutable content/recipe directory. Publish only fully prepared search views."""
import hashlib
import json
import shutil
import threading

import numpy as np
from dataclasses import asdict
from pathlib import Path

from common.news_corpus import write_json
from stage5_insight_detection.bank_analytics.library import BankLibrary
from stage5_insight_detection.bank_analytics.prepare import PreparationOptions, prepare_bank
from stage5_insight_detection.bank_analytics.search import BankAnalytics

class BankView:
    def __init__(self,parts):
        self.parts=tuple(parts)
        self.rows=[r for p in parts for r in p.rows]
        self.chunks=[r for p in parts for r in p.chunks]
        self.index={r['id']:i for i,r in enumerate(self.chunks)}
        self.available=[r for p in parts for r in p.available]; self.missing=[]
        self.reports=[r for p in parts for r in getattr(p,'reports',[])]
        self.pdf_paths={k:v for p in parts for k,v in getattr(p,'pdf_paths',{}).items()}
        self.twins={}; offset=0
        for p in parts:
            self.twins.update({i+offset:[j+offset for j in js] for i,js in p.twins.items()})
            offset+=len(p.chunks)
        self.D=np.concatenate([p.D for p in parts if p.chunks],axis=0) if self.chunks else np.empty((0,0))
        self._lexical=None; self._lexical_lock=threading.Lock(); self.bm25=self._bm25
        self.segments={};self._context_chunks=[];self._report_titles={}
        for n,part in enumerate(parts):
            self.segments.update({f'{n}:{key}':text for key,text in getattr(part,'segments',{}).items()})
            for chunk in part.chunks:
                title=f"{n}:{chunk['report']}"
                self._report_titles[title]=chunk['report']
                self._context_chunks.append(dict(chunk,report=title,
                    seg=f"{n}:{chunk['seg']}" if chunk.get('seg') is not None else None))

    def _bm25(self,query):
        if self._lexical is None:
            with self._lexical_lock:
                if self._lexical is None:
                    from stage5_insight_detection.bank_analytics.search import bm25_index
                    self._lexical=bm25_index([c['text'] for c in self.chunks])
        return self._lexical(query)

    def _top(self,query,k,task,w_dense):
        if k < 0: raise ValueError('k must be nonnegative')
        if not self.chunks or k == 0: return []
        # One query embedding and one BM25 vocabulary across ALL published reports.
        # Cached per-PDF document embeddings remain reusable; their ranks are not fused separately.
        return BankAnalytics._top(self,query,k,task,w_dense)

    def search(self,query,k=5,**kwargs):
        from stage5_insight_detection.bank_analytics.search import TASK,W_DENSE
        top=self._top(query,k,kwargs.get('task',TASK),kwargs.get('w_dense',W_DENSE))
        return [dict(self.chunks[i],score=score,pos=i) for i,score in top]

    def passages(self,query,k=3,context=2,**kwargs):
        from stage5_insight_detection.bank_analytics.search import TASK,W_DENSE
        from stage5_insight_detection.bank_analytics.chunks import passages
        top=self._top(query,k,kwargs.get('task',TASK),kwargs.get('w_dense',W_DENSE))
        scores={self.chunks[i]['id']:score for i,score in top}
        out=passages(self._context_chunks,self.segments,[i for i,_ in top],context)
        for row in out:
            row['report']=self._report_titles[row['report']]
            row['score']=max(scores[id] for id in row['ids'])
        return sorted(out,key=lambda row:(not row.get('stale',True),row['score'],row.get('date','')),reverse=True)

class Catalog:
    def __init__(self,root: Path,processor=None):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock(); self.report_locks={}; self.loaded={}; self.requested={}; self.views={}
        self.processor=processor or self._process
        path=self.root/'current.json'
        self.current=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

    def _process(self,folder,options,models):
        import importlib.util
        missing=[name for name in ('langchain_text_splitters','bm25s','Stemmer') if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError('Install requirements.txt before PDF processing; missing: '+', '.join(missing))
        prepared=prepare_bank(options,models,pdf_dir=folder)
        if prepared.report['status'] not in ('complete','disabled'):
            raise RuntimeError('PDF preparation incomplete: '+json.dumps(prepared.report,default=str))
        bank=BankAnalytics(ocr_dir=folder/'ocr',index_new=False)
        bank.reports=[dict(file=f.file,title=f.report,date=f.date or '') for f in prepared.library.fragments]
        bank.reports=list({r['file']:r for r in bank.reports}.values())
        bank.available=[r['title']+' ('+r['date']+')' for r in bank.reports]
        bank.missing=[]
        bank.pdf_paths={p.name:p for p in folder.glob('*.pdf')}
        return prepared.library,bank

    def prepare(self,pdf,options,models,seed_dir=None):
        digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
        recipe=str(getattr(getattr(models,'settings',None),'agent_model','default'))+str(getattr(getattr(models,'settings',None),'vision_model','default'))
        key=hashlib.sha256((pdf.name+digest+'api-pdf-v1'+recipe+json.dumps(asdict(options),sort_keys=True)).encode()).hexdigest()
        with self.lock:
            ticket=self.requested.get(pdf.name,0)+1; self.requested[pdf.name]=ticket
            lock=self.report_locks.setdefault(key,threading.Lock())
        with lock:
            folder=self.root/'reports'/key; folder.mkdir(parents=True,exist_ok=True)
            target=folder/pdf.name
            if not target.exists():
                shutil.copy2(pdf,target)
                if seed_dir is not None:
                    for name in ('ocr','ocr_full','charts','tables','text'):
                        source=Path(seed_dir)/name
                        if source.exists():
                            for asset in source.glob(pdf.stem+'*'):
                                if asset.is_file():
                                    (folder/name).mkdir(exist_ok=True)
                                    shutil.copy2(asset,folder/name/asset.name)
            cached=(folder/'ready.json').exists()
            if key not in self.loaded:
                library,bank=self.processor(folder,options,models)
                self.loaded[key]=(library,bank)
                write_json(folder/'ready.json',dict(file=pdf.name,key=key,digest=digest,options=asdict(options)))
            with self.lock:
                # Serialize only the pointer update; OCR/embedding work remains concurrent.
                published=self.requested[pdf.name]==ticket
                if published:
                    current=dict(self.current); current[pdf.name]=key
                    write_json(self.root/'current.json',current); self.current=current
            return dict(file=pdf.name,key=key,cached=cached,published=published,fragments=len(self.loaded[key][0].fragments))

    def snapshot(self,models):
        with self.lock:
            selected=dict(self.current)
        version=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
        with self.lock:
            if version in self.views: return self.views[version]
        parts=[]; fragments=[]
        for filename,key in sorted(selected.items()):
            if key not in self.loaded:
                manifest=json.loads((self.root/'reports'/key/'ready.json').read_text(encoding='utf-8'))
                # Historical recipes without a mode used all; do not reinterpret them via today's environment.
                options=PreparationOptions(**({'chart_mode':'all'} | manifest['options']))
                with self.lock:
                    lock=self.report_locks.setdefault(key,threading.Lock())
                with lock:
                    if key not in self.loaded:
                        self.loaded[key]=self.processor(self.root/'reports'/key,options,models)
            lib,bank=self.loaded[key]; fragments.extend(lib.fragments);parts.append(bank)
        version=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest()
        snapshot=(version,BankLibrary(fragments),BankView(parts))
        with self.lock:
            return self.views.setdefault(version,snapshot)
