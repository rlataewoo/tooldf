"""
C2 (vocal + nonvocal overlay) Tool-SFT training-data builder.

Input: label/audio_caption/c2_*.jsonl
    {"file_path": ..., "category": "c2",
     "combo": "real_real"|"real_fake"|"fake_real"|"fake_fake",
     "category_fg": "speech"|"singing",
     "category_bg": "music"|"sound",
     "label_fg": "real"|"fake", "label_bg": "real"|"fake",
     "caption_fg": str, "caption_bg": str,
     "voice_start_time": float, "voice_end_time": float,
     "duration": float, ...}
    Note: voice_start_time/voice_end_time were already shifted by fg_start_time
          in the compose_c2.py stage, so they are relative to the total audio.

Output: label/tooldf_sft/<name>.jsonl  (JSONL, or a JSON array with --indent)

Every sample is a single vocal segment + a full-length bg (music | sound):
  <think> audio_understanding (vocal + bg)
          + plan (source_separator -> vocal det -> bg det) </think>
    -> tool_call source_separator -> tool_response
    -> tool_call vocal detector (vocals_path) -> tool_response
    -> tool_call music|sound_deepfake_detector (background_path) -> tool_response
  -> assistant: description (summarizing both results)
                + answer (fake if either source is fake)

Usage:
    python preprocess/stage3/prepare_tooldf_sft_c2.py \
        --input label/audio_caption/c2_train.jsonl \
        --output label/tooldf_sft/c2_train_sft.jsonl
"""

import argparse
import json
import os
import random


# ─────────────────────────────────────────────
# Tool schemas (kept identical to c3)
# ─────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        'type': 'function',
        'function': {
            'name': 'source_separator',
            'description': 'Separate vocals from background in an audio file.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'audio_path': {'type': 'string', 'description': 'Path to the input audio file'},
                    'vocals_path': {'type': 'string', 'description': 'Output path for the separated vocals'},
                    'background_path': {'type': 'string', 'description': 'Output path for the separated background'},
                },
                'required': ['audio_path', 'vocals_path', 'background_path'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'speech_deepfake_detector',
            'description': 'Detect whether a speech segment is real or fake.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'input_path': {'type': 'string', 'description': 'Original audio path'},
                    'vocals_path': {'type': 'string', 'description': 'Separated vocals path from source_separator. Provide this if source_separator was called; otherwise omit.'},
                    'start': {'type': 'number', 'description': 'Segment start time in seconds'},
                    'end': {'type': 'number', 'description': 'Segment end time in seconds'},
                },
                'required': ['input_path', 'start', 'end'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'singing_deepfake_detector',
            'description': 'Detect whether a singing voice segment is real or fake.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'input_path': {'type': 'string', 'description': 'Original audio path'},
                    'vocals_path': {'type': 'string', 'description': 'Separated vocals path from source_separator. Provide this if source_separator was called; otherwise omit.'},
                    'start': {'type': 'number', 'description': 'Segment start time in seconds'},
                    'end': {'type': 'number', 'description': 'Segment end time in seconds'},
                },
                'required': ['input_path', 'start', 'end'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'music_deepfake_detector',
            'description': 'Detect whether a background music segment is real or fake.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'input_path': {'type': 'string', 'description': 'Original audio path'},
                    'background_path': {'type': 'string', 'description': 'Separated background path from source_separator. Provide this if source_separator was called; otherwise omit.'},
                    'start': {'type': 'number', 'description': 'Segment start time in seconds'},
                    'end': {'type': 'number', 'description': 'Segment end time in seconds'},
                },
                'required': ['input_path', 'start', 'end'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'sound_deepfake_detector',
            'description': 'Detect whether a general sound/environmental audio segment is real or fake.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'input_path': {'type': 'string', 'description': 'Path to the input audio file'},
                    'background_path': {'type': 'string', 'description': 'Separated background path from source_separator. Provide this if source_separator was called; otherwise omit.'},
                    'start': {'type': 'number', 'description': 'Segment start time in seconds'},
                    'end': {'type': 'number', 'description': 'Segment end time in seconds'},
                },
                'required': ['input_path', 'start', 'end'],
            },
        },
    },
]


# ─────────────────────────────────────────────
# Per-category prompt components
# ─────────────────────────────────────────────
VOCAL_CONFIG = {
    'speech': {
        'tool_name': 'speech_deepfake_detector',
        'plan_detect': 'Speech detected. Call speech_deepfake_detector',
        'description_subject': 'speech segment',
    },
    'singing': {
        'tool_name': 'singing_deepfake_detector',
        'plan_detect': 'Singing voice detected. Call singing_deepfake_detector',
        'description_subject': 'singing voice segment',
    },
}

BG_CONFIG = {
    'music': {
        'tool_name': 'music_deepfake_detector',
        'understanding_note': 'Background music with melodic/instrumental content detected throughout.',
        'plan_sep_reason': 'Background music overlaps with vocals. Call source_separator first.',
        'plan_detect': 'Background music. Call music_deepfake_detector on the separated background.',
        'description_subject': 'background music',
    },
    'sound': {
        'tool_name': 'sound_deepfake_detector',
        'understanding_note': 'Environmental sound/noise detected throughout.',
        'plan_sep_reason': 'Background sound overlaps with vocals. Call source_separator first.',
        'plan_detect': 'Background sound. Call sound_deepfake_detector on the separated background.',
        'description_subject': 'background sound segment',
    },
}


def _fmt(t: float) -> float:
    return round(float(t), 2)


def _score_for(label: str, rng: random.Random) -> float:
    if label == 'real':
        return round(rng.uniform(0.85, 0.99), 2)
    return round(rng.uniform(0.01, 0.15), 2)


def build_sample(entry: dict, rng: random.Random) -> dict:
    """Turn one c2 entry into one Tool-SFT sample (one vocal + one bg detector)."""
    audio_path = entry['file_path']

    cat_fg = entry['category_fg']
    cat_bg = entry['category_bg']
    label_fg, label_bg = entry['label_fg'], entry['label_bg']
    cap_fg = entry.get('caption_fg', '').strip()
    cap_bg = entry.get('caption_bg', '').strip()
    vs, ve = _fmt(entry['voice_start_time']), _fmt(entry['voice_end_time'])
    bg_s, bg_e = _fmt(0.0), _fmt(entry['duration'])

    cfg_fg = VOCAL_CONFIG[cat_fg]
    cfg_bg = BG_CONFIG[cat_bg]

    score_fg = _score_for(label_fg, rng)
    score_bg = _score_for(label_bg, rng)

    # ─────────── <think> ───────────
    audio_understanding = (
        f'[{vs}(s)-{ve}(s)] {cap_fg} '
        f'[{bg_s}(s)-{bg_e}(s)] {cap_bg} '
        f'{cfg_bg["understanding_note"]}'
    )

    plan = (
        f'1. {cfg_bg["plan_sep_reason"]}\n'
        f'2. [{vs}(s)-{ve}(s)] {cfg_fg["plan_detect"]} on the separated vocals.\n'
        f'3. [{bg_s}(s)-{bg_e}(s)] {cfg_bg["plan_detect"]}'
    )

    think = (
        f'<think>\n<audio_understanding>{audio_understanding}</audio_understanding>\n'
        f'<plan>\n{plan}\n</plan>\n</think>'
    )

    # ─────────── tool_call / tool_response ───────────
    messages = [
        {'role': 'user', 'content': '<audio>Is this audio real or fake?'},
        {'role': 'assistant', 'content': think},
    ]

    sep_call = json.dumps({
        'name': 'source_separator',
        'arguments': {
            'audio_path': 'input_audio',
            'vocals_path': 'vocals.wav',
            'background_path': 'background.wav',
        },
    }, ensure_ascii=False)
    sep_resp = json.dumps({
        'model': 'source_separator',
        'vocals_path': 'vocals.wav',
        'background_path': 'background.wav',
    }, ensure_ascii=False)
    messages.append({'role': 'tool_call', 'content': sep_call})
    messages.append({'role': 'tool_response', 'content': sep_resp})

    # vocal segment (single)
    fg_args = {'input_path': 'input_audio', 'vocals_path': 'vocals.wav',
               'start': vs, 'end': ve}
    tool_call_fg = json.dumps({'name': cfg_fg['tool_name'], 'arguments': fg_args},
                              ensure_ascii=False)
    tool_resp_fg = json.dumps({
        'model': cfg_fg['tool_name'],
        'start': vs, 'end': ve,
        'prediction': label_fg, 'score': score_fg,
    }, ensure_ascii=False)
    messages.append({'role': 'tool_call', 'content': tool_call_fg})
    messages.append({'role': 'tool_response', 'content': tool_resp_fg})

    # bg detector
    bg_args = {'input_path': 'input_audio', 'background_path': 'background.wav',
               'start': bg_s, 'end': bg_e}
    tool_call_bg = json.dumps({'name': cfg_bg['tool_name'], 'arguments': bg_args},
                              ensure_ascii=False)
    tool_resp_bg = json.dumps({
        'model': cfg_bg['tool_name'],
        'start': bg_s, 'end': bg_e,
        'prediction': label_bg, 'score': score_bg,
    }, ensure_ascii=False)
    messages.append({'role': 'tool_call', 'content': tool_call_bg})
    messages.append({'role': 'tool_response', 'content': tool_resp_bg})

    # ─────────── final assistant ───────────
    overall = 'fake' if ('fake' in (label_fg, label_bg)) else 'real'

    description = (
        f'The {cfg_fg["description_subject"]} [{vs}(s)-{ve}(s)] is detected as '
        f'{label_fg}, with a realness score of {score_fg}. '
        f'The {cfg_bg["description_subject"]} [{bg_s}(s)-{bg_e}(s)] is detected as '
        f'{label_bg}, with a realness score of {score_bg}.'
    )
    final_assistant = f'<description>{description}</description>\n<answer>{overall}</answer>'
    messages.append({'role': 'assistant', 'content': final_assistant})

    return {
        'tools': TOOL_SCHEMAS,
        'audios': [audio_path],
        'messages': messages,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='c2 AudioCaption JSONL path')
    p.add_argument('--output', required=True, help='output path (.jsonl or .json)')
    p.add_argument('--limit', type=int, default=None, help='max number of samples')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--indent', type=int, default=None,
                   help='write as a JSON array with this indent. '
                        'When omitted, writes JSONL (one sample per line).')
    args = p.parse_args()

    rng = random.Random(args.seed)

    samples = []
    skipped = 0
    combo_counts = {}
    fg_counts = {}
    bg_counts = {}
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get('category') != 'c2':
                skipped += 1
                continue
            if entry.get('category_fg') not in VOCAL_CONFIG:
                skipped += 1
                continue
            if entry.get('category_bg') not in BG_CONFIG:
                skipped += 1
                continue
            if entry.get('label_fg') not in ('real', 'fake') or \
               entry.get('label_bg') not in ('real', 'fake'):
                skipped += 1
                continue
            samples.append(build_sample(entry, rng))
            combo = entry.get('combo', 'unknown')
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
            fg_key = f'{entry["category_fg"]}_{entry["label_fg"]}'
            fg_counts[fg_key] = fg_counts.get(fg_key, 0) + 1
            bg_key = f'{entry["category_bg"]}_{entry["label_bg"]}'
            bg_counts[bg_key] = bg_counts.get(bg_key, 0) + 1
            if args.limit and len(samples) >= args.limit:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        if args.indent is None:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        else:
            json.dump(samples, f, ensure_ascii=False, indent=args.indent)

    answers = [s['messages'][-1]['content'].split('<answer>')[-1].split('</answer>')[0]
               for s in samples]
    real_ct = sum(1 for a in answers if a == 'real')
    fake_ct = sum(1 for a in answers if a == 'fake')
    print(f'[done] wrote {len(samples)} samples → {args.output}')
    print(f'       real={real_ct}, fake={fake_ct}, skipped={skipped}')
    for c in ('real_real', 'real_fake', 'fake_real', 'fake_fake'):
        if c in combo_counts:
            print(f'       combo {c:<9} = {combo_counts[c]}')
    for k in ('speech_real', 'speech_fake', 'singing_real', 'singing_fake'):
        if k in fg_counts:
            print(f'       fg    {k:<13} = {fg_counts[k]}')
    for k in ('music_real', 'music_fake', 'sound_real', 'sound_fake'):
        if k in bg_counts:
            print(f'       bg    {k:<10} = {bg_counts[k]}')


if __name__ == '__main__':
    main()
