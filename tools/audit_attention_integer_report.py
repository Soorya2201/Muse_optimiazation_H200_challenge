"""Validate every raw event bit and unchanged predeclared gate after extraction."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];TRACK=ROOT/'muse_attention_decode'
out=json.loads((TRACK/f'PK2_HYBRID_REPORT_{args.job}.json').read_text())
receipt=json.loads((TRACK/f'run_receipts/{args.job}.json').read_text())
assert receipt['source_sha256']=='881f9a61d5f9f6343505680eae30b092e23f1c79090dee436102d03b388ad252'
assert hashlib.sha256((ROOT/receipt['source']).read_bytes()).hexdigest()==receipt['source_sha256']
report=out['diagnostic']
assert report['variant_names']=={'0':'parent','1':'integer25','2':'integer375'}
assert report['gates']==dict(parent_global=.01,parent_row=.05,fallback_global=.15,fallback_row=.75)
all_us=[];max_aa=0.;max_global=max_row=0.
for record in report['cases']:
    raw=record['event_ms_float32_bits']
    assert len(raw)==4 and all(len(row)==9 for row in raw)
    us=[[struct.unpack('<f',struct.pack('<I',v))[0]*1000 for v in row] for row in raw]
    assert all(math.isfinite(v) and v>0 for row in us for v in row)
    ratios=[statistics.median(math.sqrt(a*c)/b for a,b,c in zip(us[0],us[i],us[-1])) for i in (1,2)]
    aa=statistics.median(a/b for a,b in zip(us[0],us[-1]))
    assert [round(v,6) for v in ratios]==record['r']
    assert round(aa,6)==record['aa']
    max_aa=max(max_aa,abs(aa-1))
    for e in record['e']+record['fresh']:
        assert e[0]<.01 and e[1]<.05
        max_global=max(max_global,e[0]);max_row=max(max_row,e[1])
    all_us.append(dict(pad=record['pad'],batch=record['b'],seed=record['seed'],
                       microseconds=us,median_ratios=ratios,duplicate_ratio=aa))
if report['status']=='complete':
    assert receipt['summary']['passed'] and len(report['nearby'])==6
    assert len(report['cases'])==16
    assert {(r['pad'],r['b'],r['seed']) for r in report['cases']}==set(itertools.product(range(4),(16,32),(3209,7909)))
    assert all(a[1]==0 for a in report['resources'].values())
audit=dict(job=args.job,scope='Independent diagnostic; unchanged timed parent point is not a candidate speedup.',
    diagnostic_status=report['status'],raw_event_float32_values=len(all_us)*36,
    every_raw_float32_value_and_ratio_reproduced=True,
    max_parent_global=max_global,max_parent_row=max_row,max_duplicate_offset=max_aa,
    unchanged_gates_confirmed=True,resources=report['resources'],decoded_event_groups=all_us,
    fixed_composition_results=out['compositions'],error=report.get('error'))
target=TRACK/f'INTEGER_EXP_AUDIT_{args.job}.json';assert not target.exists()
target.write_text(json.dumps(audit,indent=2)+'\n')
print('AUDIT',report['status'],'groups',len(all_us),'event values',len(all_us)*36,
      'AAmax',max_aa,'maxglobal',max_global,'maxrow',max_row)
print('RESOURCES',report['resources'])
for row in out['summaries']:print(row)
if out['compositions']:print('BEST',max(out['compositions'],key=lambda x:x['worst_model']))
