"""Stage build artifacts into `docs/` for GitHub Pages.

    python scripts/publish_pages.py [--output-dir output] [--docs-dir docs]

Copies the newest report to `docs/index.html`, keeps a dated copy alongside
it, and writes the results snapshot for anyone who wants the raw numbers.

This script only *stages* files. Pushing is the workflow's job, and it pushes
to a single-commit `pages-live` branch that gets amended and force-pushed
each run, so a ~650 KB daily artifact never accumulates in git history. The
sibling stock model added roughly 90 MB a day before adopting that scheme and
tripped GitHub's repository size limits.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def newest(pattern):
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default='output')
    ap.add_argument('--docs-dir', default='docs')
    args = ap.parse_args(argv)

    html = newest(os.path.join(args.output_dir, 'currency_screener_*.html'))
    if not html:
        print('No report found in %s — run analyze_currencies.py first.'
              % args.output_dir, file=sys.stderr)
        return 2

    os.makedirs(args.docs_dir, exist_ok=True)
    stamp = os.path.basename(html).replace('currency_screener_', '').replace('.html', '')

    shutil.copy2(html, os.path.join(args.docs_dir, 'index.html'))
    shutil.copy2(html, os.path.join(args.docs_dir, os.path.basename(html)))
    print('staged %s -> %s/index.html' % (html, args.docs_dir))

    results = os.path.join(args.output_dir, 'results_%s.json' % stamp)
    if os.path.exists(results):
        shutil.copy2(results, os.path.join(args.docs_dir, 'results.json'))
        print('staged %s -> %s/results.json' % (results, args.docs_dir))

    # A tiny manifest so the published site is introspectable without
    # parsing the 650 KB page.
    manifest = {'run_date': stamp,
                'staged_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'report': 'index.html',
                'results': 'results.json'}
    try:
        with open(results, encoding='utf-8') as f:
            payload = json.load(f)
        manifest['count'] = payload.get('count')
        manifest['provenance'] = payload.get('provenance')
    except (OSError, ValueError):
        pass
    with open(os.path.join(args.docs_dir, 'manifest.json'), 'w',
              encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)

    # .nojekyll stops GitHub Pages' Jekyll pass from touching the output.
    open(os.path.join(args.docs_dir, '.nojekyll'), 'w').close()

    total = sum(os.path.getsize(os.path.join(args.docs_dir, f))
                for f in os.listdir(args.docs_dir))
    print('docs/ ready — %d files, %.1f MB'
          % (len(os.listdir(args.docs_dir)), total / 1e6))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
