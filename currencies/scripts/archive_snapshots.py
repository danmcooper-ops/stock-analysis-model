"""Push the daily check's history to the `data/snapshots` branch.

    python scripts/archive_snapshots.py [--home DIR] [--no-push]

`check_site.py` keeps one snapshot per day under `~/CurrencyScreener/history/
<date>/{results,manifest}.json`. That folder is the only copy: `pages-live`
holds just the latest publish, and the cloud routine's containers kept
nothing. This mirrors every day in the history folder onto `data/snapshots`
as `<date>/results.json` and `<date>/manifest.json`, one commit per run.

It builds the commit with git plumbing (a private index, `mktree` via
`write-tree`, `commit-tree`) on top of `origin/data/snapshots`, so the
checkout's working tree, index and current branch are never touched. Files
already on the branch with the same content are left alone, so a day with
nothing new makes no commit. Days pruned from the local history stay on the
branch: the branch is the long-term archive.

A push rejected because the branch moved is retried once from the new tip.
Exit 0 when the branch is current (pushed or nothing to do), 1 otherwise.
Stdlib only.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HOME = os.path.expanduser('~/CurrencyScreener')
BRANCH = 'data/snapshots'
REMOTE_REF = 'refs/remotes/origin/' + BRANCH
FILES = ('results.json', 'manifest.json')

# Used only when the checkout has no git identity of its own, so a fresh Mac
# can commit without one and no personal address lands in a public repo.
BOT_NAME = 'Currency screener check'
BOT_EMAIL = 'danmcooper-ops@users.noreply.github.com'


def history_files(history_dir):
    """[(branch path, local path)] for every archived day, oldest first."""
    try:
        days = sorted(n for n in os.listdir(history_dir)
                      if re.fullmatch(r'\d{4}-\d{2}-\d{2}', n))
    except OSError:
        return []
    out = []
    for day in days:
        for name in FILES:
            p = os.path.join(history_dir, day, name)
            if os.path.isfile(p):
                out.append(('%s/%s' % (day, name), p))
    return out


def commit_message(days):
    if not days:
        return 'Snapshot: nothing new'
    if len(days) == 1:
        return 'Snapshot: %s' % days[0]
    return 'Snapshot: %s (%d days, %s .. %s)' % (days[-1], len(days),
                                                 days[0], days[-1])


class Git:
    def __init__(self, repo, env=None):
        self.repo = repo
        self.env = dict(os.environ, **(env or {}))

    def __call__(self, *args, input=None, check=True):
        r = subprocess.run(['git', '-C', self.repo, *args], input=input,
                           capture_output=True, text=True, env=self.env,
                           timeout=120)
        if check and r.returncode != 0:
            raise RuntimeError('git %s: %s' % (' '.join(args),
                                               r.stderr.strip()))
        return r


def identity_env(git):
    env = {}
    for key, var, fallback in (('user.name', 'NAME', BOT_NAME),
                               ('user.email', 'EMAIL', BOT_EMAIL)):
        if not git('config', key, check=False).stdout.strip():
            env['GIT_AUTHOR_' + var] = env['GIT_COMMITTER_' + var] = fallback
    return env


def remote_tip(git):
    """Fetch the branch; its tip sha, or None if it does not exist yet."""
    git('fetch', '--quiet', 'origin',
        '+refs/heads/%s:%s' % (BRANCH, REMOTE_REF), check=False)
    r = git('rev-parse', '--verify', '--quiet', REMOTE_REF + '^{commit}',
            check=False)
    return r.stdout.strip() or None


def build_commit(git, parent, files):
    """Commit `files` over `parent`; (sha, new days) or (None, []) if unchanged."""
    fd, index = tempfile.mkstemp(prefix='snapshots-index-')
    os.close(fd)
    os.remove(index)  # git wants to create it itself
    g = Git(git.repo, dict(git.env, GIT_INDEX_FILE=index))
    try:
        if parent:
            g('read-tree', parent)
        changed = set()
        for path, local in files:
            blob = g('hash-object', '-w', '--', local).stdout.strip()
            cur = g('ls-files', '-s', '--', path).stdout.split()
            if cur and cur[1] == blob:
                continue
            g('update-index', '--add', '--cacheinfo',
              '100644,%s,%s' % (blob, path))
            changed.add(path.split('/', 1)[0])
        if not changed:
            return None, []
        tree = g('write-tree').stdout.strip()
        days = sorted(changed)
        args = ['commit-tree', tree, '-m', commit_message(days)]
        if parent:
            args += ['-p', parent]
        return g(*args).stdout.strip(), days
    finally:
        if os.path.exists(index):
            os.remove(index)


def archive(repo, history_dir, push=True, log=print):
    git = Git(repo)
    git.env.update(identity_env(git))
    files = history_files(history_dir)
    if not files:
        log('archive: no history in %s; nothing to archive' % history_dir)
        return 0
    for attempt in (1, 2):
        parent = remote_tip(git)
        sha, days = build_commit(git, parent, files)
        if sha is None:
            log('archive: %s already holds every day in %s' % (BRANCH,
                                                               history_dir))
            return 0
        if not push:
            log('archive: built %s (%s), not pushed'
                % (sha[:8], commit_message(days)))
            return 0
        r = git('push', '--quiet', 'origin', '%s:refs/heads/%s' % (sha, BRANCH),
                check=False)
        if r.returncode == 0:
            git('update-ref', REMOTE_REF, sha, check=False)
            log('archive: pushed %s to %s (%s)'
                % (sha[:8], BRANCH, commit_message(days)))
            return 0
        err = r.stderr.strip()
        if attempt == 1 and re.search(r'non-fast-forward|fetch first|rejected',
                                      err):
            log('archive: %s moved during the push; retrying from its new tip'
                % BRANCH)
            continue
        log('archive: push to %s FAILED: %s' % (BRANCH, err))
        return 1
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--home', default=os.environ.get('CURRENCY_CHECK_HOME',
                                                     DEFAULT_HOME),
                    help='work directory holding history/')
    ap.add_argument('--no-push', action='store_true',
                    help='build the commit but do not push it')
    args = ap.parse_args(argv)
    try:
        return archive(REPO_ROOT, os.path.join(args.home, 'history'),
                       push=not args.no_push)
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        print('archive: FAILED: %s' % e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
