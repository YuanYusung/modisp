#!/usr/bin/env python3
"""Inventory public source and distributions; exclude research/runtime caches."""
import argparse,difflib,hashlib,json,re,tarfile,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path);p.add_argument('--out',type=Path);p.add_argument('--dist',type=Path,action='append');a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.out or root/'release';out.mkdir(parents=True,exist_ok=True)
allowed_dirs=['cpp','python','tests','docs','examples','tools','scripts','.github','cuda-plugin']
allowed_files=['README.md','pyproject.toml','CMakeLists.txt','LICENSE','THIRD_PARTY_NOTICES.md','CITATION.cff','CHANGELOG.md','.gitignore']
skip={'__pycache__','.numba-cache','.pytest_cache','build','dist'}
files=[]
for directory in allowed_dirs:
    for f in (root/directory).rglob('*'):
        if f.is_file() and not any(s in skip or s.startswith('._') or s=='.DS_Store' for s in f.relative_to(root).parts) and f.suffix not in ('.pyc','.so','.dylib'):files.append(f)
files += [root/f for f in allowed_files if (root/f).exists()]
files=sorted(set(files));records=[];diff=[]
baseline_pkg=None
if a.baseline and (a.baseline/'python').is_dir():
    packages=[p for p in (a.baseline/'python').iterdir() if p.is_dir() and p.name!='__pycache__']
    if len(packages)==1:baseline_pkg=packages[0]
for f in files:
    rel=f.relative_to(root).as_posix();data=f.read_bytes();origin='new project implementation or documentation';license_name='BSD-3-Clause';old=None
    if rel.startswith('cpp/') and a.baseline:old=a.baseline/'src'/f.relative_to(root/'cpp')
    if rel.startswith('python/modisp/') and baseline_pkg is not None:old=baseline_pkg/Path(rel).relative_to('python/modisp')
    if rel=='python/modisp/_legacy.py' and baseline_pkg is not None:old=baseline_pkg/'__init__.py'
    if rel=='tools/qed_reference_adapter.cpp' and a.baseline:old=a.baseline/'scripts/reference_driver.cpp'
    if old is None and a.baseline and (a.baseline/rel).exists():old=a.baseline/rel
    # Modern snapshots keep the same public layout. Prefer their exact path
    # over legacy src/package migration mappings when that path is present.
    if a.baseline and (a.baseline/rel).is_file():old=a.baseline/rel
    original=b''
    if old is not None and old.exists():
        original=old.read_bytes();origin='retained/adapted from supplied baseline'
    if original!=data and (f.suffix in ('.py','.cpp','.hpp','.cu','.cuh','.toml','.md','.cff','.yml') or f.name=='CMakeLists.txt'):
        before='baseline/'+old.relative_to(a.baseline).as_posix() if old is not None and old.exists() else '/dev/null'
        diff+=difflib.unified_diff(original.decode().splitlines(True),data.decode().splitlines(True),fromfile=before,tofile='modisp/'+rel)
    if rel in ['cpp/common/dunkin.hpp','cpp/common/physics.hpp']:origin+='; normalized P-SV/SH formulations adapted from CPS/disba; complete notices retained';license_name='BSD-3-Clause AND MIT'
    if rel=='cpp/common/refinement.hpp':origin+='; Chandrupatla interpolation test adapted from SciPy with full BSD notice retained'
    if rel.startswith('cuda-plugin/cpp/'):origin='exact copy of shared project cpp tree for self-contained plugin sdist'
    if rel in ['cuda-plugin/cpp/common/dunkin.hpp','cuda-plugin/cpp/common/physics.hpp']:license_name='BSD-3-Clause AND MIT'
    if 'NVIDIA_CUDA_12.6_EULA' in rel:origin='unmodified installed CUDA 12.6 toolkit EULA';license_name='NVIDIA-CUDA-12.6'
    if rel.startswith('tests/data/') and f.suffix=='.npz':origin='self-generated synthetic inputs and independently computed numeric references'
    records.append(dict(path=rel,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),origin=origin,license=license_name))
(out/'SOURCE_PROVENANCE.json').write_text(json.dumps(records,indent=2)+'\n');(out/'SOURCE_DIFF.patch').write_text(''.join(diff))
# Compare every public file against pinned author implementation blobs.
lock=json.loads((root/'tools/external_reference.lock.json').read_text());author={x['sha'] for x in lock if x['repository']=='pan3rock/QEDispInv'}
for f in files:
    b=f.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    if blob in author:raise SystemExit(f'External QEDispInv source leaked into public tree: {f}')
inventories=[]
for folder in a.dist or [root/'dist',root/'cuda-plugin/dist']:
    for archive in sorted(folder.glob('*')):
        if archive.suffix=='.whl':
            with zipfile.ZipFile(archive) as z:names=z.namelist()
        elif archive.name.endswith('.tar.gz'):
            with tarfile.open(archive) as t:names=t.getnames()
        else:continue
        bad=[n for n in names if any(x in n for x in ['/research/','/.venv','__pycache__','.numba-cache','libqed_reference','QEDispInv/src/','/build/'])]
        if bad:raise SystemExit(f'Unwanted distribution files in {archive}: {bad[:8]}')
        label=archive.relative_to(root).as_posix() if archive.is_relative_to(root) else str(archive.resolve())
        inventories.append(dict(archive=label,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=names))
(out/'DISTRIBUTION_INVENTORY.json').write_text(json.dumps(inventories,indent=2)+'\n')
version=re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(),re.M).group(1)
archive=out/f'modisp_v{version}_public_source.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for f in files:t.add(f,arcname=f'modisp_v{version}/'+f.relative_to(root).as_posix(),recursive=False)
print(json.dumps(dict(public_files=len(records),distributions=len(inventories),qed_implementation_files=0,source_archive=str(archive))))
