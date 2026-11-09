"""Apply each serving diff to an isolated index at the official base.

Checks packaging and Python syntax; does not run or time SGLang/GPU code.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = '297e54fd105e974cdf401e5b4d20253a22c16f90'
parser = argparse.ArgumentParser()
parser.add_argument('patches', nargs='+', type=Path)
parser.add_argument('--sglang', type=Path, default=ROOT/'serving/current/sglang')
parser.add_argument('--output', type=Path, default=ROOT/'serving/PATCH_MANIFEST_2026-11-09.json')
args = parser.parse_args()
assert not args.output.exists(), 'Choose a new report path to preserve prior evidence'
rows = []
for patch in args.patches:
    data = patch.read_bytes()
    assert data and len(data) <= 400000, ('patch byte limit', patch, len(data))
    added = [line[1:] for line in data.splitlines() if line.startswith(b'+') and not line.startswith(b'+++')]
    assert all(len(line) <= 400 for line in added), ('added line limit', patch)
    assert all(x not in data for x in (b'GIT binary patch', b'rename from ', b'old mode ', b'new mode '))
    with tempfile.TemporaryDirectory(prefix='muse_serving_verify_') as tmp:
        env = os.environ.copy()
        env['GIT_INDEX_FILE'] = str(Path(tmp)/'index')

        def git(*cmd):
            return subprocess.check_output(['git', '-C', str(args.sglang), *cmd], env=env)

        git('read-tree', BASE)
        git('apply', '--cached', '--check', str(patch.resolve()))
        git('apply', '--cached', str(patch.resolve()))
        changed = git('diff', '--cached', '--name-only', '-z', BASE).decode().split('\0')
        files = []
        for path in filter(None, changed):
            parsed = PurePosixPath(path)
            assert str(parsed) == path and '..' not in parsed.parts
            assert path.startswith('python/sglang/') and not path.startswith('python/sglang/srt/model_loader/')
            assert parsed.suffix in {'.py', '.cu', '.cuh', '.h', '.hpp', '.c', '.cc', '.cpp', '.json'}
            mode = git('ls-files', '--stage', '--', path).decode().split()[0]
            assert mode == '100644', ('source file mode', path, mode)
            contents = git('show', ':'+path)
            if parsed.suffix == '.py':
                ast.parse(contents.decode(), filename=path)
            if parsed.suffix == '.json':
                json.loads(contents)
            files.append(dict(path=path, sha256=hashlib.sha256(contents).hexdigest()))
        assert files
        rows.append(dict(patch=str(patch.relative_to(ROOT)) if patch.is_absolute() else str(patch),
                         sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                         maximum_added_line_bytes=max(map(len, added), default=0),
                         applies_to_base=True, python_syntax=True, files=files))
out = dict(scope='Source patch/package verification only; no GPU, numerical, serving or throughput validation.', base=BASE, patches=rows)
args.output.parent.mkdir(exist_ok=True, parents=True)
args.output.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
