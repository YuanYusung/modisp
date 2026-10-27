#!/usr/bin/env python3
"""Optional external solid-Rayleigh reference; never imported during solving.

Fetch upstream implementations only into an explicit external cache. No author
implementation or compiled reference library belongs in the runtime package.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
QED='ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa'

def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

LOCK=json.loads(Path(__file__).with_name('external_reference.lock.json').read_text())

def verify(repo,repository):
    items=[x for x in LOCK if x['repository']==repository]
    for item in items:
        path=repo/item['path']
        if not path.is_file():return False
        data=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if blob!=item['sha']:raise SystemExit(f'External file differs from pinned GitHub blob: {path}')
    return True

def checkout(cache,name,repository,revision,fetch,source=None):
    repo=cache/name;items=[x for x in LOCK if x['repository']==repository]
    if source is not None:
        source=source.resolve()
        if not verify(source,repository):raise SystemExit(f'Missing external source files: {source}')
        for item in items:
            dest=repo/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/item['path'],dest)
    if not verify(repo,repository):
        if not fetch:raise SystemExit(f'{repo} is incomplete; use --fetch or supply verified --qed-source and --fmt-source snapshots')
        if not (repo/'.git').exists():
            repo.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','init',str(repo)],check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['git','-C',str(repo),'remote','add','origin',f'https://github.com/{repository}.git'],check=True)
        subprocess.run(['git','-C',str(repo),'fetch','--depth=1','origin',revision],check=True,timeout=60)
        subprocess.run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD'],check=True)
        if not verify(repo,repository):raise SystemExit('Pinned upstream checkout is incomplete')
    return repo,dict(reference=revision,verified_git_blobs={x['path']:x['sha'] for x in items})

p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--fetch',action='store_true');p.add_argument('--eigen',type=Path,default=Path('/usr/include/eigen3'));p.add_argument('--compiler',default=os.environ.get('CXX','c++'));p.add_argument('--qed-source',type=Path);p.add_argument('--fmt-source',type=Path);a=p.parse_args();a.cache=a.cache.resolve();a.cache.mkdir(parents=True,exist_ok=True)
if not (a.eigen/'Eigen/Dense').exists():raise SystemExit('Provide installed Eigen3 headers using --eigen; no system packages are installed by this tool')
qed,qhead=checkout(a.cache,'QEDispInv','pan3rock/QEDispInv',QED,a.fetch,a.qed_source)
fmt,fhead=checkout(a.cache,'fmt','fmtlib/fmt','11.2.0',a.fetch,a.fmt_source)
out=a.cache/('libqed_reference.dylib' if sys.platform=='darwin' else 'libqed_reference.so')
adapter=Path(__file__).with_name('qed_reference_adapter.cpp')
command=[a.compiler,'-O3','-std=c++17','-fPIC','-shared','-ffp-contract=fast','-DFMT_HEADER_ONLY',f'-I{a.eigen}',f'-I{qed/"include"}',f'-I{fmt/"include"}',*[str(qed/'src'/s) for s in ['disp.cc','secfunc.cc','toms748.cc']],str(adapter),'-o',str(out)]
subprocess.run(command,check=True)
receipt=dict(qed=qhead,fmt=fhead,library=str(out),adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest(),library_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),compiler=subprocess.check_output([a.compiler,'--version'],text=True).splitlines()[0],command=command,scope='external solid Rayleigh only; author sources and their licenses remain in this cache')
(a.cache/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
