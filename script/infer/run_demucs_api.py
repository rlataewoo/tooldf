"""
Demucs FastAPI 서버 (demucs가 설치된 conda 환경에서 실행).

실행 (단일 GPU):
    conda activate demucs
    pip install fastapi uvicorn demucs soundfile torch
    python script/infer/run_demucs_api.py --host 0.0.0.0 --port 8001 --model htdemucs

실행 (멀티 GPU, 8001~8008에 GPU 0~7 각각 배치):
    python script/infer/run_demucs_api.py --host 0.0.0.0 --port 8001 --num-gpus 8 --model htdemucs

사용 (메인 env쪽):
    POST http://localhost:8001/separate
    body: {
        "audio_path": "...",
        "vocals_path": "...",
        "background_path": "...",
        "target_sr": 16000,   # optional, default = MODEL.samplerate (htdemucs=44100)
        "mono": true          # optional, default false
    }
"""

import argparse
import os
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as AF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio

app = FastAPI()

MODEL = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SeparateRequest(BaseModel):
    audio_path: str
    vocals_path: str
    background_path: str
    target_sr: Optional[int] = None
    mono: bool = False


def _load_audio(path: str, target_sr: int, target_channels: int):
    wav, sr = torchaudio.load(path)
    wav = convert_audio(wav, sr, target_sr, target_channels)
    return wav


@app.on_event('startup')
def _startup():
    global MODEL
    name = os.environ.get('DEMUCS_MODEL', 'htdemucs')
    print(f'[demucs_server] loading model: {name} on {DEVICE}')
    MODEL = get_model(name).to(DEVICE)
    MODEL.eval()
    print('[demucs_server] ready.')


@app.get('/health')
def health():
    return {'status': 'ok', 'device': DEVICE, 'sources': MODEL.sources if MODEL else None}


@app.post('/separate')
def separate(req: SeparateRequest):
    if not os.path.exists(req.audio_path):
        raise HTTPException(status_code=400, detail=f'audio not found: {req.audio_path}')

    wav = _load_audio(req.audio_path, MODEL.samplerate, MODEL.audio_channels)
    ref = wav.mean(0, keepdim=True)
    wav_in = (wav - ref.mean()) / wav.std().clamp(min=1e-8)

    with torch.no_grad():
        sources = apply_model(MODEL, wav_in[None].to(DEVICE), device=DEVICE, progress=False)[0]
    sources = sources * wav.std() + ref.mean()

    src_names = MODEL.sources  # e.g., ['drums','bass','other','vocals']
    idx_vocals = src_names.index('vocals')
    vocals = sources[idx_vocals].cpu()
    background = (sources.sum(0) - sources[idx_vocals]).cpu()

    out_sr = req.target_sr or MODEL.samplerate

    def _post(x: torch.Tensor) -> torch.Tensor:
        if req.mono and x.shape[0] > 1:
            x = x.mean(0, keepdim=True)
        if out_sr != MODEL.samplerate:
            x = AF.resample(x, MODEL.samplerate, out_sr)
        return x

    vocals = _post(vocals)
    background = _post(background)

    os.makedirs(os.path.dirname(os.path.abspath(req.vocals_path)) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(req.background_path)) or '.', exist_ok=True)
    torchaudio.save(req.vocals_path, vocals, out_sr)
    torchaudio.save(req.background_path, background, out_sr)

    return {
        'model': 'source_separator',
        'status': 'completed',
        'audio_path': req.audio_path,
        'sample_rate': out_sr,
        'channels': vocals.shape[0],
        'output': {
            'vocals_separated_path': req.vocals_path,
            'background_separated_path': req.background_path,
        },
    }


def _launch_multi(args):
    import subprocess
    import signal
    import sys

    procs = []
    for i in range(args.num_gpus):
        gpu_id = args.gpu_offset + i
        port = args.port + i
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            '--host', args.host,
            '--port', str(port),
            '--num-gpus', '1',
        ]
        if args.model:
            cmd += ['--model', args.model]
        p = subprocess.Popen(cmd, env=env)
        procs.append((p, port, gpu_id))
        print(f'[demucs_server] launched pid={p.pid} port={port} CUDA_VISIBLE_DEVICES={gpu_id}')

    def _stop(*_):
        for p, _port, _gpu in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    exit_code = 0
    try:
        for p, port, gpu_id in procs:
            rc = p.wait()
            if rc != 0:
                print(f'[demucs_server] worker port={port} gpu={gpu_id} exited rc={rc}')
                exit_code = rc
    finally:
        _stop()
    sys.exit(exit_code)


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001,
                        help='single mode: port. multi mode: starting port (port..port+num_gpus-1)')
    parser.add_argument('--model', default=None, help='demucs model name (default: htdemucs)')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='if >1, spawn N child servers (one per GPU) on consecutive ports')
    parser.add_argument('--gpu-offset', type=int, default=0,
                        help='multi mode: first GPU index (CUDA_VISIBLE_DEVICES = offset..offset+N-1)')
    args = parser.parse_args()

    if args.num_gpus > 1:
        _launch_multi(args)
        return

    if args.model:
        os.environ['DEMUCS_MODEL'] = args.model
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
