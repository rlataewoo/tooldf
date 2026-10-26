"""HTTP clients for the eight deepfake detector services.

Each detector type (speech / singing / music / sound) is served by TWO
XLSR-AASIST instances:

  General (run on the original input.wav)        ports 8021..8024
  Separator-Aware (run on demucs vocals/background.wav)
                                                 ports 8025..8028

(See script/run_aasist_detector_servers.sh.)

Routing rule (per call):

  - input.wav                              -> general detector
  - vocals.wav / background.wav (exists)   -> sep-aware detector

Back-compat: if SPEECH_DETECTOR_URL / SINGING_DETECTOR_URL / ... is set, it
overrides the *general* URL for that detector (the pre-split single-URL
behavior).

All eight share the same wire format. POST /detect with:

    {"input_path": str,
     "start": float, "end": float,
     "vocals_path": str | null,        # speech/singing
     "background_path": str | null}    # music/sound

The server returns:

    {"model": <name>, "start": float, "end": float,
     "prediction": "real" | "fake", "score": float}

— that JSON dict is JSON-stringified and returned to the agent verbatim,
matching the schema expected by parse_tool_call_segments in plugin_tooldf.py.
"""

import json
import os

import requests

DETECTOR_TIMEOUT = float(os.environ.get('DETECTOR_TIMEOUT', '120'))

# General detectors operate on input.wav.
GEN_DETECTOR_URLS = {
    'speech_deepfake_detector':  os.environ.get('SPEECH_GEN_DETECTOR_URL',  'http://localhost:8021'),
    'singing_deepfake_detector': os.environ.get('SINGING_GEN_DETECTOR_URL', 'http://localhost:8022'),
    'music_deepfake_detector':   os.environ.get('MUSIC_GEN_DETECTOR_URL',   'http://localhost:8023'),
    'sound_deepfake_detector':   os.environ.get('SOUND_GEN_DETECTOR_URL',   'http://localhost:8024'),
}
# Separator-aware detectors operate on demucs outputs.
SA_DETECTOR_URLS = {
    'speech_deepfake_detector':  os.environ.get('SPEECH_SA_DETECTOR_URL',  'http://localhost:8025'),
    'singing_deepfake_detector': os.environ.get('SINGING_SA_DETECTOR_URL', 'http://localhost:8026'),
    'music_deepfake_detector':   os.environ.get('MUSIC_SA_DETECTOR_URL',   'http://localhost:8027'),
    'sound_deepfake_detector':   os.environ.get('SOUND_SA_DETECTOR_URL',   'http://localhost:8028'),
}
# Back-compat: pre-split single-URL env vars map onto the general bucket.
for _name, _env in (
    ('speech_deepfake_detector',  'SPEECH_DETECTOR_URL'),
    ('singing_deepfake_detector', 'SINGING_DETECTOR_URL'),
    ('music_deepfake_detector',   'MUSIC_DETECTOR_URL'),
    ('sound_deepfake_detector',   'SOUND_DETECTOR_URL'),
):
    _v = os.environ.get(_env)
    if _v:
        GEN_DETECTOR_URLS[_name] = _v

# Kept for back-compat with code that imported DETECTOR_URLS directly.
DETECTOR_URLS = GEN_DETECTOR_URLS


def _pick_url(name: str, payload: dict) -> str:
    # Route by whether the detector is being asked to look at a separated
    # stem (vocals/background) vs the raw input. A non-empty path that
    # points to an existing file is the signal that separation ran.
    vp = payload.get('vocals_path')
    bp = payload.get('background_path')
    uses_sep = bool((vp and os.path.exists(vp)) or (bp and os.path.exists(bp)))
    urls = SA_DETECTOR_URLS if uses_sep else GEN_DETECTOR_URLS
    return urls[name]


def _call_detector(name: str, payload: dict) -> str:
    url = _pick_url(name, payload)
    try:
        r = requests.post(f'{url}/detect', json=payload, timeout=DETECTOR_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return json.dumps({
            'model':      name,
            'start':      float(payload.get('start') or 0.0),
            'end':        float(payload.get('end') or 0.0),
            'prediction': None,
            'score':      None,
            'error':      str(e),
        })
    # Trust the server schema — it owns prediction/score/start/end.
    data.setdefault('model', name)
    return json.dumps(data)


def speech_deepfake_detector(input_path: str, start: float, end: float,
                             vocals_path: str = None, **_) -> str:
    return _call_detector('speech_deepfake_detector', {
        'input_path':  input_path,
        'start':       float(start),
        'end':         float(end),
        'vocals_path': vocals_path,
    })


def singing_deepfake_detector(input_path: str, start: float, end: float,
                              vocals_path: str = None, **_) -> str:
    return _call_detector('singing_deepfake_detector', {
        'input_path':  input_path,
        'start':       float(start),
        'end':         float(end),
        'vocals_path': vocals_path,
    })


def music_deepfake_detector(input_path: str, start: float, end: float,
                            background_path: str = None, **_) -> str:
    return _call_detector('music_deepfake_detector', {
        'input_path':      input_path,
        'start':           float(start),
        'end':             float(end),
        'background_path': background_path,
    })


def sound_deepfake_detector(input_path: str, start: float, end: float,
                            background_path: str = None, **_) -> str:
    return _call_detector('sound_deepfake_detector', {
        'input_path':      input_path,
        'start':           float(start),
        'end':             float(end),
        'background_path': background_path,
    })
