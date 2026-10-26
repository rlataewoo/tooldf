"""
Single-category Tool-SFT training-data builder for the public datasets.

Covers the vocal (speech | singing) and nonvocal (music | sound) categories.
This is for the public source datasets only — MAADD's composed C1/C2/C3 samples
are handled by prepare_tooldf_sft_c{1,2,3}.py.

Input: label/audio_caption/*.jsonl
    vocal (speech | singing):
        {"file_path": ..., "category": "speech"|"singing", "label": "real"|"fake",
         "voice_start_time": float, "voice_end_time": float, "caption": str, ...}
    nonvocal (music | sound):
        {"file_path": ..., "category": "music"|"sound", "label": "real"|"fake",
         "duration": float, "caption": str, ...}
        Note: no VAD labels here, so the segment range is set to 0 ~ duration.

Output: label/tooldf_sft/<name>.jsonl  (JSONL, or a JSON array with --indent)

Every sample is a pure single-category scenario:
  Case 1 (vocal)    /  Case 2 (nonvocal)
  <think> audio_understanding + plan </think>
  -> tool_call: <category>_deepfake_detector
  -> tool_response: prediction/score
  -> assistant: description + answer

Usage:
    python preprocess/stage3/prepare_tooldf_sft_single.py \
        --category speech \
        --input label/audio_caption/19la_dev.jsonl \
        --output label/tooldf_sft/19la_dev_speech.jsonl \
        [--limit 1000] [--seed 42]

    python preprocess/stage3/prepare_tooldf_sft_single.py \
        --category singing \
        --input label/audio_caption/ctrsvdd_dev.jsonl \
        --output label/tooldf_sft/ctrsvdd_dev_singing.jsonl

    python preprocess/stage3/prepare_tooldf_sft_single.py \
        --category music \
        --input label/audio_caption/fakemusiccaps_dev.jsonl \
        --output label/tooldf_sft/fakemusiccaps_dev_music.jsonl

    python preprocess/stage3/prepare_tooldf_sft_single.py \
        --category sound \
        --input label/audio_caption/ESDD_dev.jsonl \
        --output label/tooldf_sft/ESDD_dev_sound.jsonl
"""

import argparse
import json
import os
import random

# ─────────────────────────────────────────────
# Tool schemas (kept identical between training and inference)
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


def _fmt(t: float) -> float:
    """Round timestamps to 2 decimal places."""
    return round(float(t), 2)


def _vocal_segment(entry: dict) -> tuple:
    """Vocal entries carry VAD labels, so use the annotated voice range."""
    return _fmt(entry['voice_start_time']), _fmt(entry['voice_end_time'])


def _nonvocal_segment(entry: dict) -> tuple:
    """Nonvocal entries have no VAD labels, so span the whole clip."""
    return _fmt(0.0), _fmt(entry['duration'])


# ─────────────────────────────────────────────
# Per-category prompt components
#   segment_fn:      how to derive the (start, end) detector segment
#   required_fields: extra keys an entry must have, else it is skipped
# ─────────────────────────────────────────────
CATEGORY_CONFIG = {
    'speech': {
        'tool_name': 'speech_deepfake_detector',
        'understanding_suffix': 'No background music or noise detected. Speech only.',
        'plan_skip': 'No background.',
        'plan_detect': 'Speech detected. Call speech_deepfake_detector.',
        'description_subject': 'speech segment',
        'segment_fn': _vocal_segment,
        'required_fields': (),
    },
    'singing': {
        'tool_name': 'singing_deepfake_detector',
        'understanding_suffix': 'No background music or noise detected. Singing voice only.',
        'plan_skip': 'No background.',
        'plan_detect': 'Singing voice detected. Call singing_deepfake_detector.',
        'description_subject': 'singing voice segment',
        'segment_fn': _vocal_segment,
        'required_fields': (),
    },
    'music': {
        'tool_name': 'music_deepfake_detector',
        'understanding_suffix': 'No vocals or speech detected. Music only.',
        'plan_skip': 'No vocals.',
        'plan_detect': 'Music detected. Call music_deepfake_detector.',
        'description_subject': 'music segment',
        'segment_fn': _nonvocal_segment,
        'required_fields': ('duration',),
    },
    'sound': {
        'tool_name': 'sound_deepfake_detector',
        'understanding_suffix': 'No vocals or music detected. General/environmental sound only.',
        'plan_skip': 'No vocals.',
        'plan_detect': 'General sound detected. Call sound_deepfake_detector.',
        'description_subject': 'sound segment',
        'segment_fn': _nonvocal_segment,
        'required_fields': ('duration',),
    },
}


def build_sample(entry: dict, rng: random.Random, category: str) -> dict:
    """Turn one input entry into one Tool-SFT sample."""
    cfg = CATEGORY_CONFIG[category]
    audio_path = entry['file_path']
    label = entry['label']  # real | fake
    caption = entry.get('caption', '').strip()
    seg_start, seg_end = cfg['segment_fn'](entry)

    # score = realness probability (1=real, 0=fake)
    #   real → 0.85~0.99, fake → 0.01~0.15
    if label == 'real':
        score = round(rng.uniform(0.85, 0.99), 2)
    else:
        score = round(rng.uniform(0.01, 0.15), 2)

    # <think> block
    audio_understanding = (
        f'[{seg_start}(s)-{seg_end}(s)] {caption} '
        f'{cfg["understanding_suffix"]}'
    )
    plan = (
        f'1. {cfg["plan_skip"]} Skip source_separator.\n'
        f'2. [{seg_start}(s)-{seg_end}(s)] {cfg["plan_detect"]}'
    )
    think = f'<think>\n<audio_understanding>{audio_understanding}</audio_understanding>\n<plan>\n{plan}\n</plan>\n</think>'

    # tool_call (only the detector for this category)
    tool_call = json.dumps({
        'name': cfg['tool_name'],
        'arguments': {'input_path': 'input_audio', 'start': seg_start, 'end': seg_end},
    }, ensure_ascii=False)

    # tool_response (paths stripped for training; prediction/score only)
    tool_response = json.dumps({
        'model': cfg['tool_name'],
        'start': seg_start, 'end': seg_end,
        'prediction': label, 'score': score,
    }, ensure_ascii=False)

    # description + answer — label first, score after as supporting evidence
    description = (
        f'The {cfg["description_subject"]} is detected as {label}, with a realness score of {score}.'
    )
    final_assistant = f'<description>{description}</description>\n<answer>{label}</answer>'

    return {
        'tools': TOOL_SCHEMAS,
        'audios': [audio_path],
        'messages': [
            {'role': 'user', 'content': '<audio>Is this audio real or fake?'},
            {'role': 'assistant', 'content': think},
            {'role': 'tool_call', 'content': tool_call},
            {'role': 'tool_response', 'content': tool_response},
            {'role': 'assistant', 'content': final_assistant},
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', required=True,
                   choices=['speech', 'singing', 'music', 'sound'],
                   help='target category (speech | singing | music | sound)')
    p.add_argument('--input', required=True, help='input JSONL path')
    p.add_argument('--output', required=True, help='output path (JSONL by default)')
    p.add_argument('--limit', type=int, default=None, help='max number of samples')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--indent', type=int, default=None,
                   help='write as a JSON array with this indent. '
                        'When omitted, writes JSONL (one sample per line).')
    args = p.parse_args()

    cfg = CATEGORY_CONFIG[args.category]
    rng = random.Random(args.seed)

    samples = []
    skipped = 0
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get('category') != args.category:
                skipped += 1
                continue
            if entry.get('label') not in ('real', 'fake'):
                skipped += 1
                continue
            if any(k not in entry for k in cfg['required_fields']):
                skipped += 1
                continue
            samples.append(build_sample(entry, rng, args.category))
            if args.limit and len(samples) >= args.limit:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        if args.indent is None:
            # JSONL: one sample per line
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        else:
            # JSON array: human readable
            json.dump(samples, f, ensure_ascii=False, indent=args.indent)

    labels = [s['messages'][-1]['content'].split('<answer>')[-1].split('</answer>')[0] for s in samples]
    real_ct = sum(1 for l in labels if l == 'real')
    fake_ct = sum(1 for l in labels if l == 'fake')
    print(f'[done] category={args.category}, wrote {len(samples)} samples → {args.output}')
    print(f'       real={real_ct}, fake={fake_ct}, skipped={skipped}')


if __name__ == '__main__':
    main()
