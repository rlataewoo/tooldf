"""XLSR-AASIST deepfake detector FastAPI server.

One server per detector type (speech / singing / music / sound). Loads the
XLSR-AASIST binary classifier (add_model/XLSR_AASIST/model.py) on startup
and exposes:

    POST /detect
    body: {
        "input_path":      str,
        "start":           float,
        "end":             float,
        "vocals_path":     str | null,   # speech/singing detectors
        "background_path": str | null    # music/sound detectors
    }
    -> {"model": <type>, "start": float, "end": float,
        "prediction": "real" | "fake", "score": float, "real_score": float}

Class index 1 = real, 0 = fake. `prediction` is "real" iff p(real) >= threshold.
`score` is the softmax probability of the predicted class (parity with the
Qwen detector server). `real_score` is always p(real) for downstream use.

Launch:
    python script/infer/run_aasist_api.py \
        --detector_type speech \
        --xlsr_path  /path/to/xlsr2_300m.pt \
        --ckpt_path  /path/to/xlsr_conformer_speech.pt \
        --threshold  0.5 \
        --port 8021
"""

import argparse
import os
import sys
from typing import Optional

import librosa
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


_HERE = os.path.dirname(os.path.abspath(__file__))
_AASIST_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'plugin', 'tools'))
if _AASIST_DIR not in sys.path:
    sys.path.insert(0, _AASIST_DIR)


DETECTOR_TYPES = ('speech', 'singing', 'music', 'sound')
TARGET_SR = 16000
REAL_IDX = 1


app = FastAPI()
STATE = {
    'model':         None,
    'device':        None,
    'dtype':         torch.float32,
    'detector_type': None,
    'threshold':     0.5,
}


class DetectRequest(BaseModel):
    input_path: str
    start: float = 0.0
    end:   float = 0.0
    vocals_path:     Optional[str] = None
    background_path: Optional[str] = None


def _pick_target_path(detector_type: str, req: DetectRequest) -> str:
    """vocals/background priority per detector type, fall back to input_path."""
    if detector_type in ('speech', 'singing'):
        if req.vocals_path and os.path.exists(req.vocals_path):
            return req.vocals_path
    elif detector_type in ('music', 'sound'):
        if req.background_path and os.path.exists(req.background_path):
            return req.background_path
    return req.input_path


def _load_segment_16k_mono(src_path: str, start: float, end: float) -> torch.Tensor:
    """Load src_path[start:end] (seconds), downmix to mono, resample to 16 kHz.

    Uses librosa so the (stereo->mono mean + native_sr->16k soxr_hq) chain
    matches the convention used by the upstream XLSR-AASIST EER eval — the
    previous torchaudio path produced systematically different scores for
    stereo 44.1/48k inputs (real audio) vs mono 16k inputs (generator
    outputs), inflating false-positives on the real class.

    Returns a 1-D float tensor at TARGET_SR. An empty/inverted range falls
    back to the full file.
    """
    duration = (end - start) if (end and end > start) else None
    offset = max(0.0, float(start)) if duration is not None else 0.0
    wav, _ = librosa.load(src_path, sr=TARGET_SR, mono=True,
                          offset=offset, duration=duration)
    return torch.from_numpy(wav)  # [T] float32 @ 16 kHz


@app.get('/health')
def health():
    return {
        'status':         'ok' if STATE['model'] is not None else 'loading',
        'detector_type':  STATE['detector_type'],
        'threshold':      STATE['threshold'],
    }


@app.post('/detect')
def detect(req: DetectRequest):
    model = STATE['model']
    if model is None:
        raise HTTPException(status_code=503, detail='model not loaded')

    detector_type = STATE['detector_type']
    threshold = STATE['threshold']
    device = STATE['device']
    dtype = STATE['dtype']

    target = _pick_target_path(detector_type, req)
    if not os.path.exists(target):
        raise HTTPException(status_code=400, detail=f'audio not found: {target}')

    wav = _load_segment_16k_mono(target, req.start, req.end).to(
        device=device, dtype=dtype)              # [T] @ 16 kHz mono
    x = wav.unsqueeze(0)                         # [1, T]
    x_len = torch.tensor([x.size(1)], dtype=torch.long, device=device)

    with torch.no_grad():
        out, _ = model(x, x_len)                 # [1, 2]
        p_real = F.softmax(out.float(), dim=-1)[0, REAL_IDX].item()

    prediction = 'real' if p_real >= threshold else 'fake'

    return {
        'model':      f'{detector_type}_deepfake_detector',
        'start':      float(req.start),
        'end':        float(req.end),
        'prediction': prediction,
        'score':      float(p_real),
    }


def _build_model(args):
    """Construct XLSR_AASIST, patching SSLModel's hardcoded XLSR path."""
    import model as aasist  # from add_model/XLSR_AASIST/model.py

    xlsr_path = args.xlsr_path

    def _patched_ssl_init(self):
        torch.nn.Module.__init__(self)
        import fairseq
        m, _cfg, _task = fairseq.checkpoint_utils.load_model_ensemble_and_task([xlsr_path])
        self.model = m[0]
        self.out_dim = 1024
    aasist.SSLModel.__init__ = _patched_ssl_init

    # XLSR_AASIST.__init__ only reads `args.variable`; everything else is
    # hardcoded inside the class. Match train.py's default (variable=False).
    class _Args:
        variable = bool(args.variable)

    return aasist.XLSR_AASIST(_Args())


def _load_state_dict(model: torch.nn.Module, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict):
        for k in ('state_dict', 'model_state_dict', 'model'):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {(k[len('module.'):] if k.startswith('module.') else k): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[detector_aasist_server] missing keys ({len(missing)}): '
              f'{missing[:5]}{"..." if len(missing) > 5 else ""}')
    if unexpected:
        print(f'[detector_aasist_server] unexpected keys ({len(unexpected)}): '
              f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')


def main():
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument('--detector_type', required=True, choices=DETECTOR_TYPES)
    p.add_argument('--xlsr_path', default=None,
                   help='XLSR-2 300M pretrained .pt path (xlsr2_300m.pt). '
                        'If omitted, server boots in stub mode.')
    p.add_argument('--ckpt_path', default=None,
                   help='XLSR_AASIST binary classifier checkpoint. '
                        'If omitted, the AASIST head uses random init.')
    p.add_argument('--threshold', type=float, required=True,
                   help='Decision threshold on p(real). prediction="real" iff p_real >= threshold.')
    p.add_argument('--variable', type=int, default=1, choices=(0, 1),
                   help='SSL forward mode. 1: variable-length (uses x_len + '
                        'padding mask). Recommended for inference on arbitrary '
                        '[start,end] crops with no fixed-length pad/trim.')
    p.add_argument('--dtype', default='float32', choices=('float32', 'bfloat16', 'float16'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, required=True)
    args = p.parse_args()

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]

    STATE['detector_type'] = args.detector_type
    STATE['threshold']     = float(args.threshold)
    STATE['device']        = device
    STATE['dtype']         = dtype

    if args.xlsr_path is None:
        print('[detector_aasist_server] WARNING: --xlsr_path not provided; '
              'booting in stub mode (/detect will return 503).')
    else:
        print(f'[detector_aasist_server] building XLSR_AASIST on {device} ({args.dtype})')
        model = _build_model(args)
        if args.ckpt_path:
            print(f'[detector_aasist_server] loading ckpt: {args.ckpt_path}')
            _load_state_dict(model, args.ckpt_path)
        else:
            print('[detector_aasist_server] WARNING: --ckpt_path not provided; '
                  'classifier head is randomly initialized (dev/plumbing only).')
        model.to(device=device, dtype=dtype)
        model.eval()
        STATE['model'] = model

    print(f'[detector_aasist_server:{args.detector_type}] '
          f'threshold={args.threshold} ready on :{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
