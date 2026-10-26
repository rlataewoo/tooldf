"""HTTP client for the demucs-based source separator.

Server: script/infer/run_demucs_api.py.

URL selection (resolved per-call so callers can set DEMUCS_SHARD_ID after
import — e.g., evaluate_tooldf_oracle_rule.py wires its --shard_id here):

  1. DEMUCS_URL                 — explicit override (back-compat)
  2. DEMUCS_HOST + DEMUCS_PORT_BASE + DEMUCS_SHARD_ID
     -> {DEMUCS_HOST}:{DEMUCS_PORT_BASE + DEMUCS_SHARD_ID}
     defaults: http://localhost, 8001, 0 -> http://localhost:8001

The model only sees `{"status": "completed"}` from this tool — the actual
output paths are passed in by the caller (run_one) and read back by the
detector tools. Path uniqueness across concurrent samples is the caller's
responsibility (use a per-sample TemporaryDirectory).
"""

import json
import os

import requests

DEMUCS_TIMEOUT = float(os.environ.get('DEMUCS_TIMEOUT', '600'))


def _demucs_url() -> str:
    explicit = os.environ.get('DEMUCS_URL')
    if explicit:
        return explicit
    host = os.environ.get('DEMUCS_HOST', 'http://localhost')
    port_base = int(os.environ.get('DEMUCS_PORT_BASE', '8001'))
    shard = int(os.environ.get('DEMUCS_SHARD_ID', '0'))
    return f'{host}:{port_base + shard}'


def source_separator(audio_path: str,
                     vocals_path: str,
                     background_path: str,
                     **_) -> str:
    try:
        r = requests.post(
            f'{_demucs_url()}/separate',
            json={
                'audio_path':      audio_path,
                'vocals_path':     vocals_path,
                'background_path': background_path,
            },
            timeout=DEMUCS_TIMEOUT,
        )
        r.raise_for_status()
        status = r.json().get('status', 'completed')
    except requests.RequestException as e:
        return json.dumps({
            'model':  'source_separator',
            'status': 'failed',
            'error':  str(e),
        })
    return json.dumps({'model': 'source_separator', 'status': status})
