"""Read-only public code search delta; stores selected metadata, not source."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    r = subprocess.run(args, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('track', nargs='?', default='muse_attention_decode')
parser.add_argument('--previous', default='GLOBAL_CODE_SEARCH_2026-10-30.json')
parser.add_argument('--output', default='GLOBAL_CODE_SEARCH_2026-10-31.json')
args = parser.parse_args()
track = args.track
assert track in {'muse_attention_decode', 'muse_bf16_proj', 'muse_decode_layer',
                 'muse_moe_w4a16', 'muse_prefill_layer'}
assert Path(args.previous).name == args.previous and Path(args.output).name == args.output
old = json.loads((ROOT / track / args.previous).read_text())
old_items = {(r['repo'], r['path'], r['blob_sha']) for r in old['items']}
target = ROOT / track / args.output
assert not target.exists()
items, pages = {}, []
query = track + ' in:file language:python'
for page in range(1, 11):
    endpoint = 'search/code?' + urllib.parse.urlencode(dict(q=query, per_page=100, page=page))
    result = command(['gh', 'api', endpoint])
    pages.append(dict(page=page, total_count=result['total_count'],
                      incomplete_results=result.get('incomplete_results'), returned=len(result['items'])))
    for row in result['items']:
        assert row['repository']['private'] is False
        item = dict(repo=row['repository']['full_name'], path=row['path'],
                    blob_sha=row['sha'], html_url=row['html_url'])
        items[(item['repo'], item['path'], item['blob_sha'])] = item
    if len(result['items']) < 100:
        break
board = command(['okc', 'leaderboard', track, '--json'])
owners = {name.split('/')[0] for name, _, _ in items}
ranks = {r['team'].lower(): {k:r.get(k) for k in ('rank', 'speedup', 'job')}
         for r in board['rows']}
out = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), query=query,
    method='Paginated public GitHub code search; metadata only. Dynamic index, not exhaustive/private source access.',
    previous=args.previous, you=board['you'], ranked=board['ranked'],
    pages=pages, items=list(items.values()), added=[v for k,v in items.items() if k not in old_items],
    repository_owners={o:ranks.get(o.lower()) for o in sorted(owners)})
target.write_text(json.dumps(out, indent=2) + '\n')
print('Unique paths:', len({(r['repo'],r['path']) for r in items.values()}),
      'repositories:', len({r['repo'] for r in items.values()}), 'new path/blob entries:', len(out['added']))
print('Our rank:', board['you'])
for row in out['added']:
    print(json.dumps(row))
