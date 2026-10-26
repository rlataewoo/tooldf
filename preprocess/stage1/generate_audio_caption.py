"""
Generate audio captions using Qwen API for AudioCaption format.
v2: adds webrtcvad-based voice_start_time / voice_end_time (seconds, 2 decimals).

Usage:
    python preprocess/stage1/generate_audio_caption.py \
        --input  label/binary_sft/19la_train.json \
        --output label/audio_caption/19la_train.jsonl \
        --category speech \
        --api-url http://10.10.0.58:8901/v1/chat/completions \
        --temperature 0.3 \
        --workers 4
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import requests
import webrtcvad
from tqdm import tqdm

EXAMPLES = {
    "speech": "A male speech at a moderate pace in a calm, low-pitched tone.",
    "singing": "A female singing voice carrying a soft melody with gentle vibrato.",
    "music": "An upbeat music track featuring acoustic guitar, bass, and drums.",
    "sound": "A rainfall sound on a rooftop with occasional distant thunder.",
}

# # with few-shot example
SYSTEM_PROMPT_TEMPLATE = (
    "You are an audio captioner. Output EXACTLY ONE short English sentence "
    "(about 10-20 words) capturing only the most essential audible traits of "
    "the given {category} clip. Match the style and length of this example: "
    "{example} Rules by category: "
    "speech -> gender + delivery tone + speaking pace + pitch range "
    "(NEVER describe the acoustic environment such as quiet, noisy, or reverberant); "
    "singing -> gender + melody/style "
    "(NEVER mention instruments, accompaniment, or background music); "
    "music -> main instruments + mood or tempo; "
    "sound -> main environmental source(s). "
    "Describe only what is clearly audible; do not invent details, and do not "
    "mention background elements that are not present. Never mention deepfake, "
    "authenticity, AI, models, or tools. "
    "Output only the sentence — no prefix, label, quotes, or extra text."
)

VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = 2


def get_audio_metadata(audio_path: str) -> dict:
    """Extract duration and sample_rate via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", audio_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        duration = round(float(stream["duration"]), 2)
        sample_rate = int(stream["sample_rate"])
        return {"duration": duration, "sample_rate": sample_rate}
    except Exception as e:
        print(f"[WARN] ffprobe failed for {audio_path}: {e}", file=sys.stderr)
        return {"duration": 0.0, "sample_rate": 0}


def _frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit mono
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + frame_size <= len(audio):
        yield {
            "bytes": audio[offset:offset + frame_size],
            "timestamp": timestamp,
            "duration": duration,
        }
        timestamp += duration
        offset += frame_size


def get_voice_bounds(audio_path: str) -> tuple[float, float]:
    """Return (voice_start_time, voice_end_time) in seconds using webrtcvad.

    Start = first speech frame onset; end = last speech frame offset.
    If no speech detected, returns (0.0, 0.0).
    """
    try:
        audio, sr = librosa.load(audio_path, sr=VAD_SAMPLE_RATE, mono=True)
        pcm = (audio * 32768).astype(np.int16).tobytes()
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

        start_time = None
        end_time = None
        for frame in _frame_generator(VAD_FRAME_MS, pcm, sr):
            if vad.is_speech(frame["bytes"], sr):
                if start_time is None:
                    start_time = frame["timestamp"]
                end_time = frame["timestamp"] + frame["duration"]

        if start_time is None:
            return 0.0, 0.0
        return round(start_time, 2), round(end_time, 2)
    except Exception as e:
        print(f"[WARN] VAD failed for {audio_path}: {e}", file=sys.stderr)
        return 0.0, 0.0


def get_caption(
    audio_path: str,
    category: str,
    api_url: str,
    temperature: float = 0.5,
    max_retries: int = 3,
) -> str:
    """Call Qwen API to generate a caption for the audio file."""
    example = EXAMPLES[category]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(category=category, example=example)
    user_prompt = f"<audio>Caption this {category} clip."

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(audio_path).suffix.lstrip(".").lower()
    mime_map = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg"}
    mime = mime_map.get(ext, f"audio/{ext}")

    payload = {
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:{mime};base64,{audio_b64}"},
                    },
                ],
            },
        ],
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] API failed for {audio_path} after {max_retries} retries: {e}", file=sys.stderr)
                return ""
            wait = 2 ** attempt
            print(f"[WARN] Retry {attempt}/{max_retries} for {audio_path} (wait {wait}s): {e}", file=sys.stderr)
            time.sleep(wait)
    return ""


def process_one(entry: dict, category: str, api_url: str, path_prefix: str, temperature: float = 0.0, vad: bool = False) -> dict | None:
    """Process a single entry: metadata + optional VAD bounds + caption."""
    audio_path = entry["audios"][0]
    label = entry["messages"][1]["content"].strip()

    if not os.path.isfile(audio_path):
        print(f"[WARN] File not found: {audio_path}", file=sys.stderr)
        return None

    meta = get_audio_metadata(audio_path)
    caption = get_caption(audio_path, category, api_url, temperature=temperature)

    filename = os.path.basename(audio_path)
    file_path = f"{path_prefix}/{filename}" if path_prefix else audio_path

    result = {
        "file_path": file_path,
        "category": category,
        "label": label,
        "duration": meta["duration"],
        "sample_rate": meta["sample_rate"],
    }
    if vad:
        voice_start, voice_end = get_voice_bounds(audio_path)
        result["voice_start_time"] = voice_start
        result["voice_end_time"] = voice_end
    result["caption"] = caption
    return result


def load_done_set(output_path: str) -> set[str]:
    """Load already-processed file_paths from existing JSONL output for resume."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["file_path"])
            except json.JSONDecodeError:
                continue
    return done


def main():
    parser = argparse.ArgumentParser(description="Generate audio captions for Tool-SFT (v2 with VAD)")
    parser.add_argument("--input", required=True, help="Input binary-SFT JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--category", default="speech", choices=list(EXAMPLES.keys()))
    parser.add_argument("--api-url", default="http://10.10.0.58:8901/v1/chat/completions")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--path-prefix", default=None,
                        help="Replace audio directory with this prefix in output file_path (default: keep original path)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N entries (for testing)")
    parser.add_argument("--vad", action="store_true",
                        help="Run webrtcvad to add voice_start_time / voice_end_time fields")
    args = parser.parse_args()

    print(f"Loading input: {args.input}")
    with open(args.input, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Total entries: {len(data)}")

    if args.limit:
        data = data[: args.limit]
        print(f"Limited to first {args.limit} entries")

    done_set = load_done_set(args.output)
    if done_set:
        print(f"Resuming: {len(done_set)} already processed, skipping them")

    remaining = []
    for entry in data:
        filename = os.path.basename(entry["audios"][0])
        fp = f"{args.path_prefix}/{filename}" if args.path_prefix else entry["audios"][0]
        if fp not in done_set:
            remaining.append(entry)
    print(f"Entries to process: {len(remaining)}")

    if not remaining:
        print("Nothing to do.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    outfile = open(args.output, "a")

    success = 0
    fail = 0

    if args.workers <= 1:
        for entry in tqdm(remaining, desc="Captioning"):
            result = process_one(entry, args.category, args.api_url, args.path_prefix, args.temperature, args.vad)
            if result and result["caption"]:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()
                success += 1
            else:
                fail += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one, entry, args.category, args.api_url, args.path_prefix, args.temperature, args.vad): entry
                for entry in remaining
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Captioning"):
                result = future.result()
                if result and result["caption"]:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outfile.flush()
                    success += 1
                else:
                    fail += 1

    outfile.close()
    print(f"\nDone! success={success}, fail={fail}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
