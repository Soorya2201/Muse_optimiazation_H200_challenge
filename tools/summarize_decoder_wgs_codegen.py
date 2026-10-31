"""Summarize archived compiler diagnostics, explicitly detecting clipped SASS."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / 'muse_decode_layer/WGS_CODEGEN_SUMMARY_2026-10-31.json'
assert not target.exists(), target
rows = []
for path in sorted((ROOT / 'muse_decode_layer').glob('WGS_CODEGEN_*2026-10-31*.json.gz')):
    archive = json.load(gzip.open(path, 'rt'))
    response = archive['response']
    stderr = '\n'.join(r.get('text','') for r in response.get('stderr',[]))
    resources = re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', stderr)
    row = dict(archive=str(path.relative_to(ROOT)),captured_utc=archive['captured_utc'],
               compile_source_sha256=archive['compile_source_sha256'],code=response.get('code'),
               fixed_ntt=archive.get('fixed_ntt_diagnostic_only'),
               explicit_gencode=archive.get('explicit_gencode',False),
               stack_spill_store_spill_load_bytes=[[int(x) for x in r] for r in resources],
               registers=[int(x) for x in re.findall(r'Used (\d+) registers',stderr)],
               compiler_diagnostics=re.findall(r'C75\d\d',stderr),
               failure=('CE RangeError: Invalid string length' if 'Invalid string length' in stderr
                        else 'Invalid generic sm_90 WGMMA target' if '.target sm_90' in stderr
                        else None), devices={})
    for name,device in response.get('devices',{}).items():
        asm = device.get('asm',[])
        instructions = [r for r in asm if r.get('opcodes')]
        texts = [' '.join(r.get('text','').split()) for r in instructions]
        counts = collections.Counter(next((p for p in line.split() if not p.startswith('@')),'') for line in texts)
        clipped = any('[truncated' in r.get('text','') for r in asm)
        scales = [r for r in asm if ('5c005c00' in r.get('text','').lower() if name=='SASS'
                                   else '1543527424' in r.get('text',''))]
        row['devices'][name] = dict(rows=len(asm),truncated=clipped,
            displayed_instructions=len(instructions),displayed_scale_constant_sites=len(scales),
            scale_source_lines=dict(collections.Counter(str((r.get('source') or {}).get('line')) for r in scales)),
            displayed_opcode_counts=dict(counts),
            displayed_text_sha256=hashlib.sha256('\n'.join(texts).encode()).hexdigest(),
            scope='Displayed prefix only; not whole-kernel instruction totals' if clipped else 'All returned rows')
    rows.append(row)
report = dict(scope='Standalone compile diagnostics, no GPU timing. Fixed-NTT/internal-linkage probes differ from the full judge extension.',
    conclusions=[
        'Default probe target corrected to explicit compute_90a -> sm_90a, matching judge extension.',
        'NTT8 base and explicit-scale arms: 96 registers, zero stack/spills; PTX scale-constant sites 80 vs 32, displayed SASS sites 16 vs 16.',
        'SASS views are clipped at 5000 lines; identical displayed instruction counts are a truncation artifact, NOT whole-kernel equality.',
        'Displayed instruction text differs between arms; do not call the generated binaries identical.',
        'Register-split GS4 NTT2/8 has zero reported stack/spills; NTT4 has 16-byte frame,20-byte spill stores,16-byte spill loads.',
        'No compiler diagnostic is a measured runtime or leaderboard gain.'
    ],rows=rows,
    unarchived_transport_failure=dict(observed_utc='2026-10-31 07:00:19 UTC',
        command='base --instantiation true --binary-host --explicit-gencode',
        error='HTTP Error 502: Bad Gateway',
        provenance='Terminal observation before transport exception archival was implemented; no response/source archive exists for this call.',
        compiler_status='UNKNOWN'))
target.write_text(json.dumps(report,indent=2)+'\n')
print(target.relative_to(ROOT),len(rows),'archived calls')
