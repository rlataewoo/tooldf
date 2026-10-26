"""
C3 = speech + singing concat (non-overlap) + nonvocal (music|sound) overlay.

Option B (source-unique strict subset):
    - The bg pool merges (music union sound) and is then split into two pools,
      real and fake.
    - bg is **never reused (no-reuse)** — each sample is popped from its pool once.
    - Along the bg label axis, real : fake = 1 : 1 (stratified per c1 combo, so
      each combo is also split 50/50 internally).
    - The bg category (music vs sound) simply follows the natural ratio of the
      merged pool.
    - If the bg pool limit (`C_bg = 2 * min(|B^r|, |B^f|)`) is below the c1
      maximum, n_ff is automatically reduced so that c3 N = min(|C1|, C_bg).

Inputs:
    --speech-input:   19la_*.jsonl          (category='speech')
    --singing-input:  ctrsvdd_*.jsonl       (category='singing')
    --bg-music-input: fakemusiccaps_*.jsonl (category='music')
    --bg-sound-input: ESDD_*.jsonl          (category='sound')

Outputs:
    <output-audio-dir>/<base1>_<base2>_<bg_base>.flac   (the mix result)
    <output-jsonl>                                      (one line per C3 sample)

Length / overlay / mixing follow the existing rules:
    L_fg = len(speech+singing concat), L_bg = len(bg) @ target_sr
    - L_bg >= L_fg:
        fg_offset ~ U[0, L_bg - L_fg]
        out[fg_offset:fg_offset+L_fg] += fg;  bg stays anchored at 0
        final length = L_bg
    - L_bg < L_fg:
        bg = tile(bg, ceil(L_fg/L_bg))[:L_fg];  fg_offset = 0
        final length = L_fg

    snr ~ U[snr_min, snr_max] (default 0 to +15 dB, relative to fg)
    scale_bg = RMS(fg) / (RMS(bg) * sqrt(10^(snr/10)))
    out = fg + scale_bg * bg;  if peak > 0.99 the whole signal is scaled down.

Reproducibility: same seed + same inputs -> same result.
"""

import argparse
import json
import math
import os
import random

import numpy as np
import librosa
import soundfile as sf


POOLS = (('speech', 'real'), ('speech', 'fake'),
         ('singing', 'real'), ('singing', 'fake'))

# (combo, order) -> [ (first_cat, first_label), (second_cat, second_label) ]
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

BG_LABELS = ['real', 'fake']
BG_CATS = ['music', 'sound']


def _fmt(t: float) -> float:
    return round(float(t), 2)


def load_mono(path: str, target_sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32)


def read_jsonl(path: str) -> list:
    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def calc_needs(n_rr: int, n_rf: int, n_fr: int, n_ff: int) -> dict:
    """Per-fg-pool consumption when each combo is built with an exact 50/50 order."""
    needs = {k: 0 for k in POOLS}
    counts = {'rr': n_rr, 'rf': n_rf, 'fr': n_fr, 'ff': n_ff}
    for c, n in counts.items():
        n_sp = n // 2
        n_sg = n - n_sp
        for order, cnt in [('speech_first', n_sp), ('singing_first', n_sg)]:
            for slot in SLOT_TABLE[(c, order)]:
                needs[slot] += cnt
    return needs


def assign_bg_labels(n: int, rng: random.Random) -> list:
    """Split bg labels real/fake exactly 50/50 across n samples, then shuffle.

    When n is odd, fake gets the extra one — a sample counts as fake overall as
    soon as any of its labels is fake, so this barely moves the overall metric.
    """
    n_r = n // 2
    n_f = n - n_r
    labels = ['real'] * n_r + ['fake'] * n_f
    rng.shuffle(labels)
    return labels


def mix_with_snr(fg: np.ndarray, bg: np.ndarray, snr_db: float,
                 fg_start: int, total_len: int) -> tuple:
    """Overlay fg onto bg at offset fg_start, scaling bg to hit the target SNR."""
    fg_rms = rms(fg)
    bg_rms = rms(bg)
    snr_lin = 10.0 ** (snr_db / 10.0)
    scale = fg_rms / (bg_rms * math.sqrt(snr_lin) + 1e-12)

    out = (bg * scale).astype(np.float32)
    if len(out) < total_len:
        out = np.concatenate([out, np.zeros(total_len - len(out), dtype=np.float32)])
    out[fg_start:fg_start + len(fg)] += fg

    peak = float(np.max(np.abs(out)))
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out, float(scale), peak


def build_entry(first: dict, second: dict, order: str, combo_full: str,
                out_path: str, dur1: float, dur2: float, target_sr: int,
                bg_entry: dict, L_fg: int, L_bg_orig: int, repeat_count: int,
                fg_offset_sec: float, snr: float, total_len: int) -> dict:
    v1s = _fmt(first['voice_start_time'] + fg_offset_sec)
    v1e = _fmt(first['voice_end_time'] + fg_offset_sec)
    v2s = _fmt(second['voice_start_time'] + dur1 + fg_offset_sec)
    v2e = _fmt(second['voice_end_time'] + dur1 + fg_offset_sec)
    return {
        'file_path': out_path,
        'category': 'c3',
        'combo': combo_full,
        'order': order,
        'category_1': first['category'],
        'category_2': second['category'],
        'category_bg': bg_entry['category'],
        'label_1': first['label'],
        'label_2': second['label'],
        'label_bg': bg_entry['label'],
        'caption_1': first.get('caption', ''),
        'caption_2': second.get('caption', ''),
        'caption_bg': bg_entry.get('caption', ''),
        'voice_start_time_1': v1s,
        'voice_end_time_1': v1e,
        'voice_start_time_2': v2s,
        'voice_end_time_2': v2e,
        'duration_1': _fmt(dur1),
        'duration_2': _fmt(dur2),
        'duration_fg': _fmt(L_fg / target_sr),
        'duration_bg_orig': _fmt(L_bg_orig / target_sr),
        'bg_repeat_count': int(repeat_count),
        'fg_start_time': _fmt(fg_offset_sec),
        'snr_db': round(float(snr), 4),
        'duration': _fmt(total_len / target_sr),
        'sample_rate': target_sr,
        'source_path_1': first['file_path'],
        'source_path_2': second['file_path'],
        'source_path_bg': bg_entry['file_path'],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speech-input', required=True)
    p.add_argument('--singing-input', required=True)
    p.add_argument('--bg-music-input', required=True)
    p.add_argument('--bg-sound-input', required=True)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--output-audio-dir', default='c3')
    p.add_argument('--target-sr', type=int, default=16000)
    p.add_argument('--snr-min', type=float, default=0.0)
    p.add_argument('--snr-max', type=float, default=15.0)
    p.add_argument('--seed', type=int, default=42)
    # Internal salt so the fg pairs differ even when reusing c1's seed.
    # (c1 is equivalent to salt=0, so --seed-salt 0 reproduces c1's ordering.)
    p.add_argument('--seed-salt', type=int, default=10007,
                   help='c3-only seed offset; 0 gives the same fg ordering as c1. '
                        'The default (10007) is a large prime chosen to produce '
                        'fg pairs different from c1.')
    # Force per-combo counts (auto when None; n_ff is reduced to fit the bg cap)
    p.add_argument('--n-rr', type=int, default=None)
    p.add_argument('--n-rf', type=int, default=None)
    p.add_argument('--n-fr', type=int, default=None)
    p.add_argument('--n-ff', type=int, default=None)
    p.add_argument('--skip-existing', action='store_true',
                   help='do not rewrite existing flac files (refresh meta only)')
    args = p.parse_args()

    effective_seed = args.seed + args.seed_salt
    rng = random.Random(effective_seed)

    # ─────────── fg pool (speech/singing) ───────────
    speech = [e for e in read_jsonl(args.speech_input)
              if e.get('category') == 'speech']
    singing = [e for e in read_jsonl(args.singing_input)
               if e.get('category') == 'singing']
    pools = {
        ('speech', 'real'): [e for e in speech if e['label'] == 'real'],
        ('speech', 'fake'): [e for e in speech if e['label'] == 'fake'],
        ('singing', 'real'): [e for e in singing if e['label'] == 'real'],
        ('singing', 'fake'): [e for e in singing if e['label'] == 'fake'],
    }
    for pool in pools.values():
        rng.shuffle(pool)

    sizes = {k: len(v) for k, v in pools.items()}
    sr_n = sizes[('speech', 'real')]
    sf_n = sizes[('speech', 'fake')]
    gr_n = sizes[('singing', 'real')]
    gf_n = sizes[('singing', 'fake')]

    # ─────────── bg pool (merged real / fake) ───────────
    music = [e for e in read_jsonl(args.bg_music_input)
             if e.get('category') == 'music']
    sound = [e for e in read_jsonl(args.bg_sound_input)
             if e.get('category') == 'sound']

    src_sizes = {
        ('music', 'real'): sum(1 for e in music if e['label'] == 'real'),
        ('music', 'fake'): sum(1 for e in music if e['label'] == 'fake'),
        ('sound', 'real'): sum(1 for e in sound if e['label'] == 'real'),
        ('sound', 'fake'): sum(1 for e in sound if e['label'] == 'fake'),
    }

    bg_pool_real = ([e for e in music if e['label'] == 'real']
                    + [e for e in sound if e['label'] == 'real'])
    bg_pool_fake = ([e for e in music if e['label'] == 'fake']
                    + [e for e in sound if e['label'] == 'fake'])

    if not bg_pool_real or not bg_pool_fake:
        raise ValueError(
            f'bg pool empty: |B^r|={len(bg_pool_real)}, |B^f|={len(bg_pool_fake)}'
        )

    rng.shuffle(bg_pool_real)
    rng.shuffle(bg_pool_fake)
    bg_cap = 2 * min(len(bg_pool_real), len(bg_pool_fake))

    # ─────────── decide sample counts ───────────
    K_default = min(sr_n, gr_n) // 2
    n_rr = args.n_rr if args.n_rr is not None else K_default
    n_rf = args.n_rf if args.n_rf is not None else K_default
    n_fr = args.n_fr if args.n_fr is not None else K_default

    if args.n_ff is not None:
        n_ff = args.n_ff
    else:
        # n_ff candidate 1: the fg pool limit
        needs_pre_ff = calc_needs(n_rr, n_rf, n_fr, 0)
        sf_left = sf_n - needs_pre_ff[('speech', 'fake')]
        gf_left = gf_n - needs_pre_ff[('singing', 'fake')]
        n_ff_fg_max = max(0, min(sf_left, gf_left))
        # n_ff candidate 2: the bg limit (Option B)
        n_ff_bg_max = max(0, bg_cap - (n_rr + n_rf + n_fr))
        n_ff = min(n_ff_fg_max, n_ff_bg_max)

    for nm, v in [('n_rr', n_rr), ('n_rf', n_rf), ('n_fr', n_fr), ('n_ff', n_ff)]:
        if v < 0:
            raise ValueError(f'{nm} must not be negative: {v}')

    needs = calc_needs(n_rr, n_rf, n_fr, n_ff)
    for key, need in needs.items():
        have = sizes[key]
        if need > have:
            raise ValueError(
                f'fg pool {key} exhausted: need={need}, have={have}. '
                f'n_rr={n_rr}, n_rf={n_rf}, n_fr={n_fr}, n_ff={n_ff}'
            )

    n_total = n_rr + n_rf + n_fr + n_ff
    # bg labels are stratified 50/50 per combo, so each label is used at most
    # ceil(n/2) times.
    bg_label_total = {'real': 0, 'fake': 0}
    for c, n in [('rr', n_rr), ('rf', n_rf), ('fr', n_fr), ('ff', n_ff)]:
        bg_label_total['real'] += n // 2
        bg_label_total['fake'] += n - n // 2
    if bg_label_total['real'] > len(bg_pool_real):
        raise ValueError(
            f'bg real pool exhausted: need={bg_label_total["real"]}, '
            f'have={len(bg_pool_real)} (Option B no-reuse). '
            f'Lower --n-ff or enlarge the pool.'
        )
    if bg_label_total['fake'] > len(bg_pool_fake):
        raise ValueError(
            f'bg fake pool exhausted: need={bg_label_total["fake"]}, '
            f'have={len(bg_pool_fake)} (Option B no-reuse).'
        )

    iters = {k: iter(v) for k, v in pools.items()}

    def pick(cat: str, label: str) -> dict:
        key = (cat, label)
        try:
            return next(iters[key])
        except StopIteration:
            raise RuntimeError(f'fg pool exhausted: {key}')

    bg_iters = {
        'real': iter(bg_pool_real),
        'fake': iter(bg_pool_fake),
    }

    def next_bg(label: str) -> dict:
        try:
            return next(bg_iters[label])
        except StopIteration:
            raise RuntimeError(
                f'bg pool exhausted: label={label} '
                f'(Option B no-reuse violated — should be caught by pre-validation)'
            )

    # ─────────── prepare outputs & log ───────────
    os.makedirs(args.output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or '.',
                exist_ok=True)

    print(f'[fg pools] speech_real={sr_n}, speech_fake={sf_n}, '
          f'singing_real={gr_n}, singing_fake={gf_n}')
    print(f'[bg pools] music_real={src_sizes[("music","real")]}, '
          f'music_fake={src_sizes[("music","fake")]}, '
          f'sound_real={src_sizes[("sound","real")]}, '
          f'sound_fake={src_sizes[("sound","fake")]}')
    print(f'[bg combined] |B^r|={len(bg_pool_real)}, |B^f|={len(bg_pool_fake)}, '
          f'C_bg = 2·min = {bg_cap}')
    print(f'[plan] n_rr={n_rr}, n_rf={n_rf}, n_fr={n_fr}, n_ff={n_ff} '
          f'(total N={n_total})')
    print(f'[plan] fg uses: sp_real={needs[("speech","real")]}/{sr_n}, '
          f'sg_real={needs[("singing","real")]}/{gr_n}, '
          f'sp_fake={needs[("speech","fake")]}/{sf_n}, '
          f'sg_fake={needs[("singing","fake")]}/{gf_n}')
    print(f'[plan] bg uses: real={bg_label_total["real"]}/{len(bg_pool_real)}, '
          f'fake={bg_label_total["fake"]}/{len(bg_pool_fake)}')

    combo_plan = [('rr', n_rr), ('rf', n_rf), ('fr', n_fr), ('ff', n_ff)]

    written = 0
    skipped_audio = 0
    used_names = set()
    cell_counts = {(c, lbl, cat): 0
                   for c, _ in combo_plan
                   for lbl in BG_LABELS
                   for cat in BG_CATS}

    with open(args.output_jsonl, 'w') as out_f:
        for combo, n in combo_plan:
            if n == 0:
                continue

            n_sp = n // 2
            n_sg = n - n_sp
            orders = ['speech_first'] * n_sp + ['singing_first'] * n_sg
            rng.shuffle(orders)
            bg_label_seq = assign_bg_labels(n, rng)

            for order, bg_lbl in zip(orders, bg_label_seq):
                (first_cat, first_label), (second_cat, second_label) = \
                    SLOT_TABLE[(combo, order)]
                first = pick(first_cat, first_label)
                second = pick(second_cat, second_label)
                bg_entry = next_bg(bg_lbl)

                y1 = load_mono(first['file_path'], args.target_sr)
                y2 = load_mono(second['file_path'], args.target_sr)
                dur1 = len(y1) / args.target_sr
                dur2 = len(y2) / args.target_sr

                fg = np.concatenate([y1, y2]).astype(np.float32)
                bg = load_mono(bg_entry['file_path'], args.target_sr)

                L_fg = len(fg)
                L_bg_orig = len(bg)

                if L_bg_orig >= L_fg:
                    fg_offset_samples = rng.randint(0, L_bg_orig - L_fg)
                    total_len = L_bg_orig
                    bg_used = bg
                    repeat_count = 1
                else:
                    repeat_count = math.ceil(L_fg / L_bg_orig)
                    bg_used = np.tile(bg, repeat_count)[:L_fg]
                    fg_offset_samples = 0
                    total_len = L_fg

                snr = rng.uniform(args.snr_min, args.snr_max)
                mixed, _scale, _peak = mix_with_snr(
                    fg, bg_used, snr, fg_offset_samples, total_len)

                fg_offset_sec = fg_offset_samples / args.target_sr

                base1 = os.path.splitext(os.path.basename(first['file_path']))[0]
                base2 = os.path.splitext(os.path.basename(second['file_path']))[0]
                base_bg = os.path.splitext(os.path.basename(bg_entry['file_path']))[0]
                out_name = f'{base1}_{base2}_{base_bg}.flac'
                if out_name in used_names:
                    k = 2
                    while f'{base1}_{base2}_{base_bg}__{k}.flac' in used_names:
                        k += 1
                    out_name = f'{base1}_{base2}_{base_bg}__{k}.flac'
                used_names.add(out_name)
                out_path = os.path.join(args.output_audio_dir, out_name)

                if args.skip_existing and os.path.exists(out_path):
                    skipped_audio += 1
                else:
                    sf.write(out_path, mixed, args.target_sr, format='FLAC')

                entry = build_entry(
                    first, second, order, COMBO_FULL[combo], out_path,
                    dur1, dur2, args.target_sr, bg_entry,
                    L_fg, L_bg_orig, repeat_count, fg_offset_sec, snr, total_len,
                )
                out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                written += 1
                cell_counts[(combo, bg_lbl, bg_entry['category'])] += 1

                if written % 500 == 0:
                    print(f'  ... {written} written '
                          f'(combo={combo}, bg={bg_lbl}/{bg_entry["category"]})')

    # ─────────── summary ───────────
    print(f'[done] total written: {written} → {args.output_jsonl}')
    print(f'       audio dir:     {args.output_audio_dir}')
    print('[cells] (c1 combo × bg_label × bg_category)')
    for combo, _ in combo_plan:
        for lbl in BG_LABELS:
            row_cells = ', '.join(
                f'{cat}:{cell_counts[(combo, lbl, cat)]}' for cat in BG_CATS)
            print(f'  {COMBO_FULL[combo]:<9}  bg={lbl:<4}  →  {row_cells}')
    print('[fg leftover]')
    for key, need in needs.items():
        print(f'       {key[0]:7}/{key[1]:4}: used {need}/{sizes[key]} '
              f'(leftover {sizes[key] - need})')
    print('[bg leftover]')
    print(f'       real: used {bg_label_total["real"]}/{len(bg_pool_real)} '
          f'(leftover {len(bg_pool_real) - bg_label_total["real"]})')
    print(f'       fake: used {bg_label_total["fake"]}/{len(bg_pool_fake)} '
          f'(leftover {len(bg_pool_fake) - bg_label_total["fake"]})')
    if args.skip_existing:
        print(f'       skipped existing audio: {skipped_audio}')
    print(f'       seed={args.seed} (+salt {args.seed_salt} → {effective_seed}), '
          f'snr=[{args.snr_min}, {args.snr_max}] dB, target_sr={args.target_sr}')


if __name__ == '__main__':
    main()
