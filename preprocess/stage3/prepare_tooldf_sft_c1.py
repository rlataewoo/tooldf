"""
C1 (speech + singing concat) Tool-SFT training-data builder.

Input: label/audio_caption/c1_*.jsonl
    {"file_path": ..., "category": "c1",
     "combo": "real_real"|"real_fake"|"fake_real"|"fake_fake",
     "order": "speech_first"|"singing_first",
     "category_1": "speech"|"singing", "category_2": "speech"|"singing",
     "label_1": "real"|"fake", "label_2": "real"|"fake",
     "caption_1": str, "caption_2": str,
     "voice_start_time_1": float, "voice_end_time_1": float,
     "voice_start_time_2": float, "voice_end_time_2": float,
     "duration_1": float, "duration_2": float, "duration": float, ...}

Output: label/tooldf_sft/<name>.jsonl  (JSONL, or a JSON array with --indent)

Every sample is a two-vocal-segment (speech + singing) scenario:
  <think> audio_understanding (both segments) + plan (two detector calls) </think>
  -> tool_call 1 (detector for the first segment)  -> tool_response 1
  -> tool_call 2 (detector for the second segment) -> tool_response 2
  -> assistant: description (summarizing both results)
                + answer (fake if either segment is fake)

Usage:
    python preprocess/stage3/prepare_tooldf_sft_c1.py \
        --input label/audio_caption/c1_train.jsonl \
        --output label/tooldf_sft/c1_train_sft.jsonl
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


# ─────────────────────────────────────────────
# Per-segment-category prompt components
# ─────────────────────────────────────────────
SEG_CONFIG = {
    'speech': {
        'tool_name': 'speech_deepfake_detector',
        'plan_detect': 'Speech detected. Call speech_deepfake_detector.',
        'description_subject': 'speech segment',
    },
    'singing': {
        'tool_name': 'singing_deepfake_detector',
        'plan_detect': 'Singing voice detected. Call singing_deepfake_detector.',
        'description_subject': 'singing voice segment',
    },
}


def _fmt(t: float) -> float:
    return round(float(t), 2)


def _score_for(label: str, rng: random.Random) -> float:
    if label == 'real':
        return round(rng.uniform(0.85, 0.99), 2)
    return round(rng.uniform(0.01, 0.15), 2)


def build_sample(entry: dict, rng: random.Random) -> dict:
    """Turn one c1 entry into one Tool-SFT sample (two segments, two detectors)."""
    audio_path = entry['file_path']

    cat1, cat2 = entry['category_1'], entry['category_2']
    label1, label2 = entry['label_1'], entry['label_2']
    cap1 = entry.get('caption_1', '').strip()
    cap2 = entry.get('caption_2', '').strip()
    v1s, v1e = _fmt(entry['voice_start_time_1']), _fmt(entry['voice_end_time_1'])
    v2s, v2e = _fmt(entry['voice_start_time_2']), _fmt(entry['voice_end_time_2'])

    cfg1 = SEG_CONFIG[cat1]
    cfg2 = SEG_CONFIG[cat2]

    score1 = _score_for(label1, rng)
    score2 = _score_for(label2, rng)

    # <think>
    audio_understanding = (
        f'[{v1s}(s)-{v1e}(s)] {cap1} '
        f'[{v2s}(s)-{v2e}(s)] {cap2} '
        'No background music or noise detected.'
    )
    plan = (
        '1. No background music. Skip source_separator.\n'
        f'2. [{v1s}(s)-{v1e}(s)] {cfg1["plan_detect"]}\n'
        f'3. [{v2s}(s)-{v2e}(s)] {cfg2["plan_detect"]}'
    )
    think = (
        f'<think>\n<audio_understanding>{audio_understanding}</audio_understanding>\n'
        f'<plan>\n{plan}\n</plan>\n</think>'
    )

    # tool_call / tool_response (in segment order)
    tool_call_1 = json.dumps({
        'name': cfg1['tool_name'],
        'arguments': {'input_path': 'input_audio', 'start': v1s, 'end': v1e},
    }, ensure_ascii=False)
    tool_resp_1 = json.dumps({
        'model': cfg1['tool_name'],
        'start': v1s, 'end': v1e,
        'prediction': label1, 'score': score1,
    }, ensure_ascii=False)

    tool_call_2 = json.dumps({
        'name': cfg2['tool_name'],
        'arguments': {'input_path': 'input_audio', 'start': v2s, 'end': v2e},
    }, ensure_ascii=False)
    tool_resp_2 = json.dumps({
        'model': cfg2['tool_name'],
        'start': v2s, 'end': v2e,
        'prediction': label2, 'score': score2,
    }, ensure_ascii=False)

    # Final answer: fake if either segment is fake
    overall = 'fake' if (label1 == 'fake' or label2 == 'fake') else 'real'

    description = (
        f'The {cfg1["description_subject"]} [{v1s}(s)-{v1e}(s)] is detected as '
        f'{label1}, with a realness score of {score1}. '
        f'The {cfg2["description_subject"]} [{v2s}(s)-{v2e}(s)] is detected as '
        f'{label2}, with a realness score of {score2}.'
    )
    final_assistant = f'<description>{description}</description>\n<answer>{overall}</answer>'

    return {
        'tools': TOOL_SCHEMAS,
        'audios': [audio_path],
        'messages': [
            {'role': 'user', 'content': '<audio>Is this audio real or fake?'},
            {'role': 'assistant', 'content': think},
            {'role': 'tool_call', 'content': tool_call_1},
            {'role': 'tool_response', 'content': tool_resp_1},
            {'role': 'tool_call', 'content': tool_call_2},
            {'role': 'tool_response', 'content': tool_resp_2},
            {'role': 'assistant', 'content': final_assistant},
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='c1 AudioCaption JSONL path')
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
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get('category') != 'c1':
                skipped += 1
                continue
            if entry.get('category_1') not in SEG_CONFIG or \
               entry.get('category_2') not in SEG_CONFIG:
                skipped += 1
                continue
            if entry.get('label_1') not in ('real', 'fake') or \
               entry.get('label_2') not in ('real', 'fake'):
                skipped += 1
                continue
            samples.append(build_sample(entry, rng))
            combo = entry.get('combo', 'unknown')
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
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


if __name__ == '__main__':
    main()
