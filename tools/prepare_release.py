#!/usr/bin/env python3
"""Make the plugin source distribution self-contained from the shared tree."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
plugin=root/'cuda-plugin'
shutil.copytree(root/'cpp',plugin/'cpp',dirs_exist_ok=True)
for name in ['LICENSE','THIRD_PARTY_NOTICES.md']:
    shutil.copy2(root/name,plugin/name)
print('Prepared CUDA plugin source:',plugin)
