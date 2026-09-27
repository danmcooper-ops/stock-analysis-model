"""Tests for scripts/publish_report.py (R2 sync behind Cloudflare Access)."""

import datetime as dt
import hashlib
import os

import pytest

from scripts import publish_report as pr

# AWS's published SigV4 examples for S3 ("Examples: Signature Calculations
# in AWS Signature Version 4", authenticating requests with the
# Authorization header). Same credentials and clock for every example.
AK = 'AKIAIOSFODNN7EXAMPLE'
SK = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
NOW = dt.datetime(2013, 5, 24, tzinfo=dt.timezone.utc)


def _sig(headers):
    return headers['Authorization'].rsplit('Signature=', 1)[1]


def test_sigv4_get_object():
    h = pr.sign_request('GET', 'https://examplebucket.s3.amazonaws.com/test.txt', {'Range': 'bytes=0-9'},
                        pr.EMPTY_SHA256, AK, SK, region='us-east-1', now=NOW)
    assert _sig(h) == 'f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41'
    assert 'SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,' in h['Authorization']
    assert 'Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request' in h['Authorization']
    assert 'host' not in h  # requests supplies Host


def test_sigv4_put_object_with_escaped_key():
    body = b'Welcome to Amazon S3.'
    h = pr.sign_request('PUT', 'https://examplebucket.s3.amazonaws.com/test$file.text',
                        {'Date': 'Fri, 24 May 2013 00:00:00 GMT', 'x-amz-storage-class': 'REDUCED_REDUNDANCY'},
                        hashlib.sha256(body).hexdigest(), AK, SK, region='us-east-1', now=NOW)
    assert _sig(h) == '98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd'


def test_sigv4_valueless_query():
    h = pr.sign_request('GET', 'https://examplebucket.s3.amazonaws.com/?lifecycle', {},
                        pr.EMPTY_SHA256, AK, SK, region='us-east-1', now=NOW)
    assert _sig(h) == 'fea454ca298b7da1c68078a5d1bdbfbbe0d65c699e0f91ac7a200a0136783543'


def test_sigv4_list_objects_sorted_query():
    h = pr.sign_request('GET', 'https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J', {},
                        pr.EMPTY_SHA256, AK, SK, region='us-east-1', now=NOW)
    assert _sig(h) == '34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7'


def test_content_type():
    assert pr.content_type('index.html').startswith('text/html')
    assert pr.content_type('px/AAPL.json').startswith('application/json')
    assert pr.content_type('blob.bin') == 'application/octet-stream'


def test_plan_sync_uploads_changed_index_last_and_deletes_stale():
    local = {'index.html': 'a', 'details.json': 'b', 'px/AAPL.json': 'c', 'px/MSFT.json': 'd'}
    remote = {'index.html': 'old', 'details.json': 'b', 'px/AAPL.json': 'stale', 'px/GONE.json': 'x'}
    uploads, deletes = pr.plan_sync(local, remote)
    assert uploads == ['px/AAPL.json', 'px/MSFT.json', 'index.html']
    assert deletes == ['px/GONE.json']


def test_plan_sync_unchanged_is_a_noop():
    local = {'index.html': 'a', 'hist.json': 'b'}
    assert pr.plan_sync(local, dict(local)) == ([], [])


class FakeBucket:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.log = []

    def list_etags(self):
        return {k: hashlib.md5(v).hexdigest() for k, v in self.objects.items()}

    def put(self, key, body):
        self.log.append(('put', key))
        self.objects[key] = body

    def delete(self, key):
        self.log.append(('delete', key))
        self.objects.pop(key)


def _docs(tmp_path, files):
    for key, body in files.items():
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return str(tmp_path)


def test_sync_orders_index_after_sidecars_and_deletes_last(tmp_path):
    docs = _docs(tmp_path, {'index.html': b'2026-09-26', 'details.json': b'{}', 'vol/AAPL.json': b'[1]'})
    bucket = FakeBucket({'index.html': b'2026-09-25', 'details.json': b'{}', 'vol/OLD.json': b'[0]'})
    pr.sync(bucket, docs, workers=2)
    assert bucket.log[-2:] == [('put', 'index.html'), ('delete', 'vol/OLD.json')]
    assert ('put', 'details.json') not in bucket.log  # unchanged
    assert bucket.objects == {'index.html': b'2026-09-26', 'details.json': b'{}', 'vol/AAPL.json': b'[1]'}


def test_sync_dry_run_sends_nothing(tmp_path):
    docs = _docs(tmp_path, {'index.html': b'x'})
    bucket = FakeBucket({'stale.json': b'y'})
    uploads, deletes = pr.sync(bucket, docs, dry_run=True)
    assert (uploads, deletes) == (['index.html'], ['stale.json'])
    assert bucket.log == []


def test_sync_refuses_without_index(tmp_path):
    docs = _docs(tmp_path, {'details.json': b'{}'})
    with pytest.raises(RuntimeError, match='index.html'):
        pr.sync(FakeBucket(), docs)


def test_sync_upload_failure_keeps_old_index(tmp_path):
    docs = _docs(tmp_path, {'index.html': b'new', 'details.json': b'new'})
    bucket = FakeBucket({'index.html': b'old', 'details.json': b'old'})

    def boom(key, body):
        raise RuntimeError('R2 PUT failed')
    bucket.put = boom
    with pytest.raises(RuntimeError):
        pr.sync(bucket, docs, workers=1)
    assert bucket.objects['index.html'] == b'old'


class FakeResp:
    def __init__(self, status, text=''):
        self.status_code, self.text = status, text


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers, timeout, allow_redirects):
        self.calls.append(headers)
        return self.responses.pop(0)


def test_verify_live_sends_service_token_and_matches_date():
    s = FakeSession([FakeResp(200, '... 2026-09-26 ...')])
    assert pr.verify_live('https://x.workers.dev/', '2026-09-26', 'id', 'secret', session=s)
    assert s.calls[0] == {'CF-Access-Client-Id': 'id', 'CF-Access-Client-Secret': 'secret'}


def test_verify_live_fails_on_login_redirect(capsys):
    s = FakeSession([FakeResp(302)] * 2)
    assert not pr.verify_live('https://x.workers.dev/', '2026-09-26', 'id', 's', attempts=2, delay=0, session=s)
    assert 'service token was refused' in capsys.readouterr().out


def test_verify_live_retries_until_new_date():
    s = FakeSession([FakeResp(200, '2026-09-25'), FakeResp(200, '2026-09-26')])
    assert pr.verify_live('https://x/', '2026-09-26', 'id', 's', attempts=3, delay=0, session=s)


def test_main_unconfigured_exits_2(tmp_path, monkeypatch):
    for k in ('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY'):
        monkeypatch.delenv(k, raising=False)
    assert pr.main([str(tmp_path), '--rundate', '2026-09-26']) == 2


def test_local_files_uses_forward_slash_keys(tmp_path):
    docs = _docs(tmp_path, {'index.html': b'', 'px/A.json': b''})
    assert set(pr.local_files(docs)) == {'index.html', f'px{"/"}A.json'}
    assert all(os.path.isabs(p) for p in pr.local_files(docs).values())
