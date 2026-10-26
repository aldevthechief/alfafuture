"""Import checked, immutable demo snapshots from a fetched Git ref. No models or runtime-cache writes.

Usage: python scripts_import_precomputed.py [ref] [--write]
Existing bundled feeds are preserved. A row needs its frozen evidence, reader identity and resolvable citations.
Legacy news rows without the original numbered facts are excluded rather than rebuilt from today's company data.
"""
import copy
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from backend.cache_check import BLOCKS, check_row
from backend.cabinet import _read_job, _row_key
from common.evidence import EvidenceSnapshot
from common.roles import canonical_role
from stage6_insight_generation.insight_writer.insight_v2 import Insight
from stage6_insight_generation.insight_compact.compact import CompactInsight, problems as compact_problems, with_sources
from stage6_insight_generation.insight_suggestions.suggest import problems as question_problems

ROOT=Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def checked(row,company,role):
    if row.get('status')!='ready': return 'not_ready'
    if not row.get('evidence'): return 'missing_frozen_evidence'
    try:
        ins=Insight.model_validate(row['insight'])
        ev=EvidenceSnapshot.model_validate(row['evidence'])
        if (ev.company_id,canonical_role(ev.role))!=(company,role): return 'wrong_reader'
        if ins.company_id and ins.company_id!=company or canonical_role(ins.role)!=role: return 'wrong_reader'
        if not (row.get('review') or {}).get('ok'): return 'not_reviewed'
        if any(level=='fail' for level,_ in check_row(row)): return 'unresolved_source'
        known=set(ev.sources())
        if any(source not in known for block in BLOCKS for claim in row['insight'].get(block,[])
               for source in claim.get('sources',[])): return 'unfrozen_citation'
    except (ValueError,KeyError,TypeError): return 'invalid_schema'
    return None


def collect(ref):
    revision=git('rev-parse',ref).decode().strip()
    paths=git('ls-tree','-r','--name-only',revision,'data/api').decode().splitlines()
    read=lambda path:json.loads(git('show',f'{revision}:{path}'))
    groups=defaultdict(list); stats=Counter(); questions={}; compacts={}
    runs=[(path,read(path)) for path in paths if path.startswith('data/api/analyses/')]
    for path,job in sorted(runs,key=lambda x:x[1].get('created_at',''),reverse=True):
        req=job['request'];pair=(req['company_id'],req['role'])
        if req.get('user_id') or job['status'] not in ('succeeded','partial'):
            stats['private_or_unfinished_job']+=1;continue
        suggested=f'data/api/suggested/{job["id"]}.json'
        compact=f'data/api/compact/{job["id"]}.json'
        qmap=read(suggested) if suggested in paths else {}
        cmap=read(compact) if compact in paths else {}
        for n,row in enumerate(job['result']['results']):
            if reason:=checked(row,*pair): stats[reason]+=1;continue
            row=copy.deepcopy(row)
            original=row['insight'].get('id') or f'row-{n}-{row.get("news_id","bank")}'
            row['insight']['id']=original
            groups[pair].append((row,job['finished_at']))
            questions[(pair,original)]=qmap.get(original)
            compacts[(pair,original)]=cmap.get(original)
    # Analytics snapshots already embed verbatim cited facts and report quotes; reuse the cabinet's conversion.
    with tempfile.TemporaryDirectory() as folder:
        for path in paths:
            if not path.startswith('data/api/analytics_cache/'):continue
            saved=read(path);pair=(saved['company_id'],saved['role'])
            local=Path(folder)/'run.json';local.write_text(json.dumps(saved),encoding='utf-8')
            try:job=_read_job(str(local),local.stat().st_mtime_ns,path)
            except (ValueError,KeyError):stats['invalid_analytics_run']+=1;continue
            for row in job['result']['results']:
                if reason:=checked(row,*pair):stats[reason]+=1;continue
                groups[pair].append((row,job['finished_at']))
    output={};summary={}
    for pair,rows in sorted(groups.items()):
        company,role=pair;name=f'{company}_{role}';job_id=f'demo-{company}-{role}'
        target=ROOT/'backend'/'demo_cache'/f'{name}.json'
        if target.exists() and json.loads(target.read_text(encoding='utf-8')).get('precomputed_source')!=revision:
            stats['existing_feed_preserved']+=1;continue
        unique={}
        for row,stamp in rows:
            key=_row_key(row)
            if key not in unique:unique[key]=(row,stamp)
        selected=[r for r,_ in unique.values()];stamp=max(t for _,t in unique.values())
        qpath=f'data/api/suggested/{job_id}.json'
        qsaved=read(qpath) if qpath in paths else {}
        qout={};cout={}
        for row in selected:
            ins=row['insight'];ins['company_id']=company;ins['role']=role
            key=ins['id'];q=questions.get((pair,key)) or qsaved.get(key)
            if isinstance(q,list) and all(isinstance(x,str) for x in q) and not question_problems(q):qout[key]=q
            c=compacts.get((pair,key))
            if c:
                try:
                    c=with_sources(CompactInsight.model_validate(c),ins)
                    if not compact_problems(c,ins):cout[key]=c
                except (ValueError,KeyError,TypeError):pass
        output[f'{name}.json']=dict(id=job_id,kind='analysis',status='succeeded',created_at=stamp,finished_at=stamp,error=None,
            precomputed_source=revision,
            request=dict(company_id=company,role=role,user_id=None,source='both'),
            result=dict(company_id=company,role=role,cached=True,results=selected))
        if qout:output[f'suggested/{job_id}.json']=qout
        if cout:output[f'compact/{job_id}.json']=cout
        summary[name]=dict(cards=len(selected),questions=len(qout),compact=len(cout))
    supplemental={}
    # Some checked feeds predate this import. Preserve their cards, but still bring their matching saved questions.
    for target in (ROOT/'backend'/'demo_cache').glob('U_*.json'):
        job=json.loads(target.read_text(encoding='utf-8'))
        req=job['request'];pair=(req['company_id'],req['role'])
        jid=f'demo-{pair[0]}-{pair[1]}';relative=f'suggested/{jid}.json'
        source=f'data/api/{relative}'
        if relative in output or source not in paths:continue
        ids={r['insight'].get('id') for r in job['result']['results'] if not checked(r,*pair)}
        qout={key:q for key,q in read(source).items() if key in ids and isinstance(q,list)
              and all(isinstance(x,str) for x in q) and not question_problems(q)}
        if qout:
            output[relative]=qout
            supplemental[jid]=len(qout)
    manifest=dict(source=revision,feeds=summary,supplemental_questions=supplemental,excluded=dict(stats),
                  note='Frozen checked rows only. No Redis seed, opaque LLM answers, private jobs or reconstructed live facts.')
    return output,manifest


if __name__=='__main__':
    ref=next((arg for arg in sys.argv[1:] if not arg.startswith('--')),'origin/demo-precomputed')
    output,manifest=collect(ref)
    if '--write' in sys.argv:
        for name,value in output.items():
            path=ROOT/'backend'/'demo_cache'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        (ROOT/'backend'/'demo_cache'/'precomputed-manifest.json').write_text(
            json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
