# scripts/run_checkpoint.py
"""Resumable progress for one analyze_stock run date.

A full-universe run takes many hours and ``analyze_stock`` only writes its
results at the very end, so an interrupted process lost everything: on
2026-09-11 the cloud routine's container restarted 3h24m and then 12h49m into
the analysis, and no snapshot was produced for that session.

The checkpoint lives in ``output/.checkpoint/<run date>/``:

``meta.json``
    The fingerprint the progress was recorded under: the run date, every
    analysis-affecting CLI option, and a hash of the pipeline's Python
    source. A different fingerprint means the saved progress would not be
    what this run computes, so it is discarded.
``phase1.jsonl``
    One line per ticker the Phase-1 screen deliberately dropped (below the
    market-cap floor, spread too low, no data from any source). A resumed run
    skips them without a fetch. Qualifying tickers are not recorded: Phase 2
    needs their fetched data in memory, so a resume fetches them again.
``phase2.pkl``
    A pickle stream with one ``(ticker, row, skip_detail)`` record per
    finished Phase-2 ticker. Pickle rather than JSON so a resumed row is the
    same Python object graph (tuples, NaN, numpy scalars) the uninterrupted
    run would have produced. A torn final record from a kill mid-write is
    ignored.

Recording never raises: a checkpoint failure costs resumability, never the
run. ``clear()`` removes the directory once the run's outputs are written.
"""

import hashlib
import json
import logging
import os
import pickle
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join('output', '.checkpoint')

# CLI options that change which tickers are analysed or how, and so must
# match for saved progress to be reusable. Everything else (thread count,
# the screen skip cache, --no-resume itself) only changes how fast a run is.
FINGERPRINT_ARGS = ('universe', 'mcap_min', 'min_spread', 'input', 'validation',
                    'tickers', 'macro', 'prices_dir')
CODE_DIRS = ('data', 'models', 'scripts')


def code_version(root=REPO_ROOT, dirs=CODE_DIRS):
    """SHA-256 over the pipeline's Python source (paths and contents)."""
    h = hashlib.sha256()
    for d in dirs:
        base = os.path.join(root, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(x for x in dirnames if x != '__pycache__')
            for fn in sorted(filenames):
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                h.update(os.path.relpath(path, root).encode('utf-8'))
                try:
                    with open(path, 'rb') as f:
                        h.update(f.read())
                except OSError:
                    continue
    return h.hexdigest()


def fingerprint(run_date, args, code=None):
    """The dict a checkpoint must match to be resumed."""
    opts = {}
    for k in FINGERPRINT_ARGS:
        v = getattr(args, k, None)
        opts[k] = sorted(v) if isinstance(v, (list, tuple)) else v
    return {'run_date': str(run_date), 'args': opts,
            'code': code if code is not None else code_version()}


class RunCheckpoint:
    def __init__(self, run_date, fp, root=DEFAULT_ROOT):
        self.dir = os.path.join(root, str(run_date))
        self.fingerprint = fp
        self._screened_out = {}
        self._phase2 = {}
        self._writes_ok = True
        self.resumed = False
        meta_path = os.path.join(self.dir, 'meta.json')
        try:
            with open(meta_path, encoding='utf-8') as f:
                saved = json.load(f)
        except (OSError, ValueError):
            saved = None
        if saved is not None and saved.get('fingerprint') == fp:
            self._load()
            self.resumed = bool(self._screened_out or self._phase2)
        else:
            if saved is not None:
                logger.warning("checkpoint %s was recorded under different options or code; "
                               "discarding it", self.dir)
            self._reset()

    # -- setup ------------------------------------------------------------
    def _reset(self):
        try:
            shutil.rmtree(self.dir, ignore_errors=True)
            os.makedirs(self.dir, exist_ok=True)
            with open(os.path.join(self.dir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump({'fingerprint': self.fingerprint,
                           'created_at': datetime.now().isoformat(timespec='seconds')}, f, indent=1)
        except OSError as e:
            logger.warning("checkpoint %s could not be created (%s); this run is not resumable",
                           self.dir, e)
            self._writes_ok = False

    def _load(self):
        try:
            with open(os.path.join(self.dir, 'phase1.jsonl'), encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # a torn last line
                    self._screened_out[rec['t']] = rec
        except OSError:
            pass
        try:
            with open(os.path.join(self.dir, 'phase2.pkl'), 'rb') as f:
                while True:
                    try:
                        ticker, row, skip = pickle.load(f)
                    except EOFError:
                        break
                    except Exception as e:
                        logger.warning("checkpoint %s: stopped reading Phase-2 records at a "
                                       "damaged entry (%s)", self.dir, e)
                        break
                    self._phase2[ticker] = (row, skip)
        except OSError:
            pass

    # -- Phase 1 ------------------------------------------------------------
    def screened_out(self, ticker):
        return ticker in self._screened_out

    def record_screened_out(self, ticker, group=None):
        if not self._writes_ok or ticker in self._screened_out:
            return
        rec = {'t': ticker, 'grp': group}
        self._screened_out[ticker] = rec
        try:
            with open(os.path.join(self.dir, 'phase1.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError as e:
            self._write_failed(e)

    # -- Phase 2 ------------------------------------------------------------
    def phase2_record(self, ticker):
        """``(row, skip_detail)`` saved for *ticker*, or None when not done."""
        return self._phase2.get(ticker)

    def record_phase2(self, ticker, row, skip_detail=None):
        if not self._writes_ok:
            return
        try:
            blob = pickle.dumps((ticker, row, skip_detail), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning("checkpoint: %s row could not be serialised (%s); it will be "
                           "recomputed if the run resumes", ticker, e)
            return
        try:
            with open(os.path.join(self.dir, 'phase2.pkl'), 'ab') as f:
                f.write(blob)
                f.flush()
                os.fsync(f.fileno())
            self._phase2[ticker] = (row, skip_detail)
        except OSError as e:
            self._write_failed(e)

    # -- lifecycle ----------------------------------------------------------
    def counts(self):
        return {'screened_out': len(self._screened_out), 'phase2': len(self._phase2)}

    def clear(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        parent = os.path.dirname(self.dir)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass

    def _write_failed(self, e):
        logger.warning("checkpoint %s: write failed (%s); further progress is not "
                       "checkpointed", self.dir, e)
        self._writes_ok = False
