"""Read-only GitHub account inventory; emit an apply_patch containing the audit.

Run: python3 tools/audit_ranked_public_repos.py muse_moe_w4a16 | apply_patch
No files are written by this script. Public metadata is a discovery screen,
not proof that an unlabelled/private submission does not exist.
"""
import concurrent.futures
import datetime
import json
from pathlib import Path
import re
import subprocess
import sys

SIGNALS = re.compile(r'muse|open.?kernel|\bokc\b|h200|kernel.?challenge|sglang|w4a16|nvfp4', re.I)
REPO_CACHE = {}

def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr.strip()[-400:])
    return json.loads(p.stdout)

def inspect(row):
    result = {k: row.get(k) for k in ('rank', 'team', 'speedup', 'job')}
    cached = REPO_CACHE.get(row['team'])
    if cached is not None:
        result.update(cached)
        return result
    try:
        pages = command(['gh', 'api', '--paginate', '--slurp',
                         f'users/{row["team"]}/repos?per_page=100&type=owner'])
        repos = [r for page in pages for r in page]
        matches = []
        for repo in repos:
            fields = [repo['name'], repo.get('description') or '', ' '.join(repo.get('topics', []))]
            if SIGNALS.search(' '.join(fields)):
                matches.append({k: repo.get(k) for k in
                                ('full_name', 'html_url', 'description', 'fork', 'pushed_at')})
        result.update(public_repo_count=len(repos), matches=matches, status='screened')
    except Exception as error:
        result.update(status='unresolved', error=str(error))
    return result

if __name__ == '__main__':
    track = sys.argv[1]
    if not re.fullmatch(r'muse_[a-z0-9_]+', track):
        raise SystemExit('Invalid track')
    if len(sys.argv) > 2:
        previous = json.loads(Path(sys.argv[2]).read_text())
        captured = datetime.datetime.fromisoformat(previous['captured_utc'])
        if (datetime.datetime.now(datetime.timezone.utc) - captured).total_seconds() > 3600:
            raise SystemExit('Repository cache is older than one hour; refresh it')
        for account in previous['accounts']:
            if account['status'] == 'screened':
                repo_captured = account.get('repo_metadata_captured_utc', previous['captured_utc'])
                repo_age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(repo_captured)
                if repo_age.total_seconds() > 3600:
                    continue  # A fresh outer snapshot must not renew an older repository cache.
                REPO_CACHE[account['team']] = {
                    'status': 'screened', 'public_repo_count': account['public_repo_count'],
                    'matches': account['matches'],
                    'repo_metadata_captured_utc': repo_captured}
    board = command(['okc', 'leaderboard', track, '--json'])
    rows = [row for row in board['rows'] if isinstance(row.get('rank'), int)]
    collected = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(inspect, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            collected.append(future.result())
            if len(collected) % 100 == 0:
                print(f'Public account screen: {len(collected)}/{len(rows)}', file=sys.stderr, flush=True)
    result = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'track': track, 'you': board['you'], 'ranked': board['ranked'],
              'accounts_checked': len(collected),
              'method': 'Paginated public owner repositories; name/description/topic keyword screen. Optional repository snapshots reused for up to one hour with capture timestamp; ranks always fresh.',
              'limits': 'Private, renamed, unlabelled or unindexed submissions may be missed. Matches need source review.',
              'accounts': sorted(collected, key=lambda x: x['rank'])}
    target = f'{track}/RANKED_GITHUB_REPO_SCREEN.json'
    print('*** Begin Patch')
    print(f'*** Add File: {target}')
    print('\n'.join('+' + line for line in json.dumps(result, indent=2).splitlines()))
    print('*** End Patch')
