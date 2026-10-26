"""
C1 (speech + singing) concatenated audio dataset builder.

Inputs:
    --speech-input:  19la_*.jsonl    (category='speech')
    --singing-input: ctrsvdd_*.jsonl (category='singing')
    Both must come from the same split.

Outputs:
    <output-audio-dir>/<base1>_<base2>.flac   (the two clips concatenated in time)
    <output-jsonl>                            (one line per C1 sample, post-merge meta)

Sampling rules (no reuse):
    - 4 combos (label_1, label_2), ordered by time:
        real_real, real_fake, fake_real, fake_fake
    - **Every source sample is used at most once.**
    - Within each combo, speech_first : singing_first = exactly 50/50.

Default sample counts (auto, when not specified):
    K = min(|speech_real|, |singing_real|) // 2
    n_rr = n_rf = n_fr = K               (1:1:1 ratio)
    n_ff = as many fakes as remain = min(|speech_fake| - K, |singing_fake| - K)

    Note on real-pool usage (under the exact 50/50 ordering):
        sp_real = sg_real = n_rr + (n_rf + n_fr)/2 = 2K  ->  2K <= min(|sr|, |gr|)
      So the smaller real pool (usually speech_real) is effectively consumed in
      full, while the larger one may leave some samples unused (no-reuse policy).

Overrides:
    --n-rr / --n-rf / --n-fr / --n-ff force individual counts
    (errors out if a pool limit is exceeded).

Reproducibility:
    Same seed + same inputs -> same result (both pool and order shuffles are
    deterministic).
"""

import argparse
import json
import os
import random

import numpy as np
import librosa
import soundfile as sf


POOLS = (('speech', 'real'), ('speech', 'fake'),
         ('singing', 'real'), ('singing', 'fake'))

# (combo, order) -> [ (first_cat, first_label), (second_cat, second_label) ]
#   combo: 'rr' | 'rf' | 'fr' | 'ff'  (r/f shorthand for label_1, label_2)
#   order: 'speech_first' | 'singing_first'
SLOT_TABLE = {
    ('rr', 'speech_first'):  [('speech', 'real'),  ('singing', 'real')],
    ('rr', 'singing_first'): [('singing', 'real'), ('speech', 'real')],
    ('rf', 'speech_first'):  [('speech', 'real'),  ('singing', 'fake')],
    ('rf', 'singing_first'): [('singing', 'real'), ('speech', 'fake')],
    ('fr', 'speech_first'):  [('speech', 'fake'),  ('singing', 'real')],
    ('fr', 'singing_first'): [('singing', 'fake'), ('speech', 'real')],
    ('ff', 'speech_first'):  [('speech', 'fake'),  ('singing', 'fake')],
    ('ff', 'singing_first'): [('singing', 'fake'), ('speech', 'fake')],
}

COMBO_FULL = {'rr': 'real_real', 'rf': 'real_fake',
              'fr': 'fake_real', 'ff': 'fake_fake'}


def _fmt(t: float) -> float:
    return round(float(t), 2)


def load_mono(path: str, target_sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y


def read_jsonl(path: str) -> list:
    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_pair_entry(first: dict, second: dict, order: str, combo_full: str,
                     out_path: str, dur1: float, dur2: float,
                     target_sr: int) -> dict:
    v1s = _fmt(first['voice_start_time'])
    v1e = _fmt(first['voice_end_time'])
    v2s = _fmt(float(second['voice_start_time']) + dur1)
    v2e = _fmt(float(second['voice_end_time']) + dur1)
    return {
        'file_path': out_path,
        'category': 'c1',
        'combo': combo_full,
        'order': order,
        'category_1': first['category'],
        'category_2': second['category'],
        'label_1': first['label'],
        'label_2': second['label'],
        'caption_1': first.get('caption', ''),
        'caption_2': second.get('caption', ''),
        'voice_start_time_1': v1s,
        'voice_end_time_1': v1e,
        'voice_start_time_2': v2s,
        'voice_end_time_2': v2e,
        'duration_1': _fmt(dur1),
        'duration_2': _fmt(dur2),
        'duration': _fmt(dur1 + dur2),
        'sample_rate': target_sr,
        'source_path_1': first['file_path'],
        'source_path_2': second['file_path'],
    }


def calc_needs(n_rr: int, n_rf: int, n_fr: int, n_ff: int) -> dict:
    """Per-pool consumption when each combo is built with an exact 50/50 order."""
    needs = {k: 0 for k in POOLS}
    counts = {'rr': n_rr, 'rf': n_rf, 'fr': n_fr, 'ff': n_ff}
    for c, n in counts.items():
        n_sp = n // 2
        n_sg = n - n_sp
        for order, cnt in [('speech_first', n_sp), ('singing_first', n_sg)]:
            for slot in SLOT_TABLE[(c, order)]:
                needs[slot] += cnt
    return needs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speech-input', required=True)
    p.add_argument('--singing-input', required=True)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--output-audio-dir', default='c1')
    p.add_argument('--target-sr', type=int, default=16000)
    p.add_argument('--seed', type=int, default=42)
    # Force per-combo counts (auto when None)
    p.add_argument('--n-rr', type=int, default=None, help='number of real_real samples')
    p.add_argument('--n-rf', type=int, default=None, help='number of real_fake samples')
    p.add_argument('--n-fr', type=int, default=None, help='number of fake_real samples')
    p.add_argument('--n-ff', type=int, default=None, help='number of fake_fake samples')
    p.add_argument('--skip-existing', action='store_true',
                   help='do not rewrite existing flac files (refresh meta only)')
    args = p.parse_args()

    # ─────────────────────────────────────────────
    # Build pools
    # ─────────────────────────────────────────────
    speech = read_jsonl(args.speech_input)
    singing = read_jsonl(args.singing_input)
    speech = [e for e in speech if e.get('category') == 'speech']
    singing = [e for e in singing if e.get('category') == 'singing']

    pools = {
        ('speech', 'real'): [e for e in speech if e['label'] == 'real'],
        ('speech', 'fake'): [e for e in speech if e['label'] == 'fake'],
        ('singing', 'real'): [e for e in singing if e['label'] == 'real'],
        ('singing', 'fake'): [e for e in singing if e['label'] == 'fake'],
    }

    rng = random.Random(args.seed)
    for pool in pools.values():
        rng.shuffle(pool)

    sizes = {k: len(v) for k, v in pools.items()}
    sr_n = sizes[('speech', 'real')]
    sf_n = sizes[('speech', 'fake')]
    gr_n = sizes[('singing', 'real')]
    gf_n = sizes[('singing', 'fake')]

    # ─────────────────────────────────────────────
    # Decide sample counts
    # ─────────────────────────────────────────────
    # Default: n_rr = n_rf = n_fr = K, K = min(sr, gr) // 2
    K_default = min(sr_n, gr_n) // 2
    n_rr = args.n_rr if args.n_rr is not None else K_default
    n_rf = args.n_rf if args.n_rf is not None else K_default
    n_fr = args.n_fr if args.n_fr is not None else K_default

    # n_ff: unless given, drain as much of the remaining fake pools as possible
    if args.n_ff is not None:
        n_ff = args.n_ff
    else:
        needs_pre_ff = calc_needs(n_rr, n_rf, n_fr, 0)
        sf_left = sf_n - needs_pre_ff[('speech', 'fake')]
        gf_left = gf_n - needs_pre_ff[('singing', 'fake')]
        n_ff = max(0, min(sf_left, gf_left))

    for nm, v in [('n_rr', n_rr), ('n_rf', n_rf), ('n_fr', n_fr), ('n_ff', n_ff)]:
        if v < 0:
            raise ValueError(f'{nm} must not be negative: {v}')

    # Final pool-limit validation
    needs = calc_needs(n_rr, n_rf, n_fr, n_ff)
    for key, need in needs.items():
        have = sizes[key]
        if need > have:
            raise ValueError(
                f'pool {key} exhausted: need={need}, have={have}. '
                f'n_rr={n_rr}, n_rf={n_rf}, n_fr={n_fr}, n_ff={n_ff}'
            )

    # ─────────────────────────────────────────────
    # Picker: drain every pool by popping (no reuse)
    # ─────────────────────────────────────────────
    iters = {k: iter(v) for k, v in pools.items()}

    def pick(cat: str, label: str) -> dict:
        key = (cat, label)
        try:
            return next(iters[key])
        except StopIteration:
            raise RuntimeError(f'pool exhausted: {key} (should be caught by pre-validation)')

    # ─────────────────────────────────────────────
    # Prepare outputs
    # ─────────────────────────────────────────────
    os.makedirs(args.output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or '.',
                exist_ok=True)

    print(f'[pools] speech_real={sr_n}, speech_fake={sf_n}, '
          f'singing_real={gr_n}, singing_fake={gf_n}')
    print(f'[plan]  n_rr={n_rr}, n_rf={n_rf}, n_fr={n_fr}, n_ff={n_ff} '
          f'(total={n_rr + n_rf + n_fr + n_ff})')
    print(f'[plan]  uses: sp_real={needs[("speech","real")]}/{sr_n}, '
          f'sg_real={needs[("singing","real")]}/{gr_n}, '
          f'sp_fake={needs[("speech","fake")]}/{sf_n}, '
          f'sg_fake={needs[("singing","fake")]}/{gf_n}')

    # Fixed combo processing order: rr -> rf -> fr -> ff
    combo_plan = [('rr', n_rr), ('rf', n_rf), ('fr', n_fr), ('ff', n_ff)]

    written = 0
    skipped_audio = 0
    used_names = set()
    combo_counts = {c: 0 for c, _ in combo_plan}

    with open(args.output_jsonl, 'w') as out_f:
        for combo, n in combo_plan:
            if n == 0:
                continue

            # exact 50/50 order, seed shuffle
            n_sp = n // 2
            n_sg = n - n_sp
            orders = ['speech_first'] * n_sp + ['singing_first'] * n_sg
            rng.shuffle(orders)

            for order in orders:
                (first_cat, first_label), (second_cat, second_label) = \
                    SLOT_TABLE[(combo, order)]
                first = pick(first_cat, first_label)
                second = pick(second_cat, second_label)

                y1 = load_mono(first['file_path'], args.target_sr)
                y2 = load_mono(second['file_path'], args.target_sr)
                dur1 = len(y1) / args.target_sr
                dur2 = len(y2) / args.target_sr

                base1 = os.path.splitext(os.path.basename(first['file_path']))[0]
                base2 = os.path.splitext(os.path.basename(second['file_path']))[0]
                out_name = f'{base1}_{base2}.flac'

                # No-reuse policy holds, but guard against distinct pairs that
                # happen to produce the same base name.
                if out_name in used_names:
                    k = 2
                    while f'{base1}_{base2}__{k}.flac' in used_names:
                        k += 1
                    out_name = f'{base1}_{base2}__{k}.flac'
                used_names.add(out_name)

                out_path = os.path.join(args.output_audio_dir, out_name)

                if args.skip_existing and os.path.exists(out_path):
                    skipped_audio += 1
                else:
                    combined = np.concatenate([y1, y2])
                    sf.write(out_path, combined, args.target_sr, format='FLAC')

                entry = build_pair_entry(first, second, order, COMBO_FULL[combo],
                                         out_path, dur1, dur2, args.target_sr)
                out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                written += 1
                combo_counts[combo] += 1

                if written % 500 == 0:
                    print(f'  ... {written} written (current combo={combo})')

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    print(f'[done] total written: {written} → {args.output_jsonl}')
    print(f'       audio dir:    {args.output_audio_dir}')
    for c, _ in combo_plan:
        print(f'       {COMBO_FULL[c]:<9} = {combo_counts[c]}')
    for key, need in needs.items():
        print(f'       used {key[0]:7}/{key[1]:4}: {need}/{sizes[key]} '
              f'(leftover {sizes[key] - need})')
    if args.skip_existing:
        print(f'       skipped existing audio: {skipped_audio}')
    print(f'       seed={args.seed}, target_sr={args.target_sr}')


if __name__ == '__main__':
    main()
