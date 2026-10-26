"""
C2 = vocal (speech | singing) foreground + nonvocal (music | sound) background
     OVERLAY (SNR mix). One fg + one bg.

Option B (source-unique strict subset):
    - fg pools: vocal_real / vocal_fake (= speech union singing, per label)
    - bg pools: nonvocal_real / nonvocal_fake (= music union sound, per label)
    - **no reuse on either side**: each sample is popped from its pool once.
    - The vocal/nonvocal category split (speech vs singing, music vs sound)
      follows the natural ratio of each label pool; only the label is
      distributed deterministically.
    - 4 combos (label_fg, label_bg) in {rr, rf, fr, ff}.

Inputs:
    --speech-input:   19la_*.jsonl          (category='speech')
    --singing-input:  ctrsvdd_*.jsonl       (category='singing')
    --bg-music-input: fakemusiccaps_*.jsonl (category='music')
    --bg-sound-input: ESDD_*.jsonl          (category='sound')

Outputs:
    <output-audio-dir>/<base_fg>_<base_bg>.flac   (the overlay result)
    <output-jsonl>                                (one line per C2 sample)

Length / overlay / mixing (same as c3):
    L_fg = len(vocal) @ target_sr,  L_bg = len(nonvocal) @ target_sr
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

Sample counts (auto):
    K = min(|V^r|, |B^r|) // 2
    n_rr = n_rf = n_fr = K
    n_ff = min(|V^f| - K, |B^f| - K)

Reproducibility: same seed (+salt) + same inputs -> same result.
"""

import argparse
import json
import math
import os
import random

import numpy as np
import librosa
import soundfile as sf


COMBO_FULL = {'rr': 'real_real', 'rf': 'real_fake',
              'fr': 'fake_real', 'ff': 'fake_fake'}

# combo -> (fg_label, bg_label)
COMBO_LABELS = {'rr': ('real', 'real'),
                'rf': ('real', 'fake'),
                'fr': ('fake', 'real'),
                'ff': ('fake', 'fake')}


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


def mix_with_snr(fg: np.ndarray, bg: np.ndarray, snr_db: float,
                 fg_start: int, total_len: int) -> tuple:
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


def build_entry(fg_e: dict, bg_e: dict, combo_full: str, out_path: str,
                target_sr: int, L_fg: int, L_bg_orig: int, repeat_count: int,
                fg_offset_sec: float, snr: float, total_len: int) -> dict:
    vs = _fmt(float(fg_e['voice_start_time']) + fg_offset_sec)
    ve = _fmt(float(fg_e['voice_end_time']) + fg_offset_sec)
    return {
        'file_path': out_path,
        'category': 'c2',
        'combo': combo_full,
        'category_fg': fg_e['category'],
        'category_bg': bg_e['category'],
        'label_fg': fg_e['label'],
        'label_bg': bg_e['label'],
        'caption_fg': fg_e.get('caption', ''),
        'caption_bg': bg_e.get('caption', ''),
        'voice_start_time': vs,
        'voice_end_time': ve,
        'duration_fg': _fmt(L_fg / target_sr),
        'duration_bg_orig': _fmt(L_bg_orig / target_sr),
        'bg_repeat_count': int(repeat_count),
        'fg_start_time': _fmt(fg_offset_sec),
        'snr_db': round(float(snr), 4),
        'duration': _fmt(total_len / target_sr),
        'sample_rate': target_sr,
        'source_path_fg': fg_e['file_path'],
        'source_path_bg': bg_e['file_path'],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--speech-input', required=True)
    p.add_argument('--singing-input', required=True)
    p.add_argument('--bg-music-input', required=True)
    p.add_argument('--bg-sound-input', required=True)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--output-audio-dir', default='c2')
    p.add_argument('--target-sr', type=int, default=16000)
    p.add_argument('--snr-min', type=float, default=0.0)
    p.add_argument('--snr-max', type=float, default=15.0)
    p.add_argument('--seed', type=int, default=42)
    # Prime salt so fg/bg ordering differs from c1/c3.
    # Defaults: c1=0 (implicit), c3=10007, c2=20011.
    p.add_argument('--seed-salt', type=int, default=20011,
                   help='c2-only seed offset; guarantees a shuffle sequence '
                        'different from c1/c3')
    # Force per-combo counts (auto when None)
    p.add_argument('--n-rr', type=int, default=None)
    p.add_argument('--n-rf', type=int, default=None)
    p.add_argument('--n-fr', type=int, default=None)
    p.add_argument('--n-ff', type=int, default=None)
    p.add_argument('--skip-existing', action='store_true',
                   help='skip existing flac files (refresh meta only)')
    args = p.parse_args()

    effective_seed = args.seed + args.seed_salt
    rng = random.Random(effective_seed)

    # ─────────── fg pool (vocal: speech union singing, per label) ───────────
    speech = [e for e in read_jsonl(args.speech_input)
              if e.get('category') == 'speech']
    singing = [e for e in read_jsonl(args.singing_input)
               if e.get('category') == 'singing']

    src_fg_sizes = {
        ('speech', 'real'): sum(1 for e in speech if e['label'] == 'real'),
        ('speech', 'fake'): sum(1 for e in speech if e['label'] == 'fake'),
        ('singing', 'real'): sum(1 for e in singing if e['label'] == 'real'),
        ('singing', 'fake'): sum(1 for e in singing if e['label'] == 'fake'),
    }

    fg_pool_real = ([e for e in speech if e['label'] == 'real']
                    + [e for e in singing if e['label'] == 'real'])
    fg_pool_fake = ([e for e in speech if e['label'] == 'fake']
                    + [e for e in singing if e['label'] == 'fake'])

    if not fg_pool_real or not fg_pool_fake:
        raise ValueError(
            f'fg pool empty: |V^r|={len(fg_pool_real)}, |V^f|={len(fg_pool_fake)}'
        )

    rng.shuffle(fg_pool_real)
    rng.shuffle(fg_pool_fake)

    # ─────────── bg pool (nonvocal: music union sound, per label) ───────────
    music = [e for e in read_jsonl(args.bg_music_input)
             if e.get('category') == 'music']
    sound = [e for e in read_jsonl(args.bg_sound_input)
             if e.get('category') == 'sound']

    src_bg_sizes = {
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

    Vr, Vf = len(fg_pool_real), len(fg_pool_fake)
    Br, Bf = len(bg_pool_real), len(bg_pool_fake)

    # ─────────── decide sample counts ───────────
    K_default = min(Vr, Br) // 2
    n_rr = args.n_rr if args.n_rr is not None else K_default
    n_rf = args.n_rf if args.n_rf is not None else K_default
    n_fr = args.n_fr if args.n_fr is not None else K_default

    if args.n_ff is not None:
        n_ff = args.n_ff
    else:
        # Pool consumption (assuming n_rr = n_rf = n_fr = K):
        #   u(V^r) = n_rr + n_rf,   u(V^f) = n_fr + n_ff
        #   u(B^r) = n_rr + n_fr,   u(B^f) = n_rf + n_ff
        vf_left = Vf - n_fr
        bf_left = Bf - n_rf
        n_ff = max(0, min(vf_left, bf_left))

    for nm, v in [('n_rr', n_rr), ('n_rf', n_rf), ('n_fr', n_fr), ('n_ff', n_ff)]:
        if v < 0:
            raise ValueError(f'{nm} must not be negative: {v}')

    # Pool-limit validation
    u_vr = n_rr + n_rf
    u_vf = n_fr + n_ff
    u_br = n_rr + n_fr
    u_bf = n_rf + n_ff
    if u_vr > Vr:
        raise ValueError(f'vocal_real exhausted: need={u_vr}, have={Vr}')
    if u_vf > Vf:
        raise ValueError(f'vocal_fake exhausted: need={u_vf}, have={Vf}')
    if u_br > Br:
        raise ValueError(f'nonvocal_real exhausted: need={u_br}, have={Br}')
    if u_bf > Bf:
        raise ValueError(f'nonvocal_fake exhausted: need={u_bf}, have={Bf}')

    # ─────────── pickers ───────────
    fg_iters = {'real': iter(fg_pool_real), 'fake': iter(fg_pool_fake)}
    bg_iters = {'real': iter(bg_pool_real), 'fake': iter(bg_pool_fake)}

    def next_fg(label: str) -> dict:
        try:
            return next(fg_iters[label])
        except StopIteration:
            raise RuntimeError(f'fg pool exhausted: {label}')

    def next_bg(label: str) -> dict:
        try:
            return next(bg_iters[label])
        except StopIteration:
            raise RuntimeError(f'bg pool exhausted: {label}')

    # ─────────── prepare outputs & log ───────────
    os.makedirs(args.output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or '.',
                exist_ok=True)

    n_total = n_rr + n_rf + n_fr + n_ff
    print(f'[fg src]  speech_real={src_fg_sizes[("speech","real")]}, '
          f'speech_fake={src_fg_sizes[("speech","fake")]}, '
          f'singing_real={src_fg_sizes[("singing","real")]}, '
          f'singing_fake={src_fg_sizes[("singing","fake")]}')
    print(f'[fg pool] |V^r|={Vr}, |V^f|={Vf}')
    print(f'[bg src]  music_real={src_bg_sizes[("music","real")]}, '
          f'music_fake={src_bg_sizes[("music","fake")]}, '
          f'sound_real={src_bg_sizes[("sound","real")]}, '
          f'sound_fake={src_bg_sizes[("sound","fake")]}')
    print(f'[bg pool] |B^r|={Br}, |B^f|={Bf}')
    print(f'[plan]    n_rr={n_rr}, n_rf={n_rf}, n_fr={n_fr}, n_ff={n_ff} '
          f'(total N={n_total})')
    print(f'[plan]    fg uses: V^r={u_vr}/{Vr}, V^f={u_vf}/{Vf}')
    print(f'[plan]    bg uses: B^r={u_br}/{Br}, B^f={u_bf}/{Bf}')

    combo_plan = [('rr', n_rr), ('rf', n_rf), ('fr', n_fr), ('ff', n_ff)]

    # Build the full combo sequence, then shuffle it globally so that the
    # fg/bg categories (speech/singing/music/sound) appear in random order.
    sample_seq = []
    for combo, n in combo_plan:
        sample_seq.extend([combo] * n)
    rng.shuffle(sample_seq)

    written = 0
    skipped_audio = 0
    used_names = set()
    cell_counts = {(c, fg_cat, bg_cat): 0
                   for c in COMBO_FULL
                   for fg_cat in ('speech', 'singing')
                   for bg_cat in ('music', 'sound')}
    combo_counts = {c: 0 for c in COMBO_FULL}

    with open(args.output_jsonl, 'w') as out_f:
        for combo in sample_seq:
            fg_lbl, bg_lbl = COMBO_LABELS[combo]
            fg_e = next_fg(fg_lbl)
            bg_e = next_bg(bg_lbl)

            fg = load_mono(fg_e['file_path'], args.target_sr)
            bg = load_mono(bg_e['file_path'], args.target_sr)

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

            base_fg = os.path.splitext(os.path.basename(fg_e['file_path']))[0]
            base_bg = os.path.splitext(os.path.basename(bg_e['file_path']))[0]
            out_name = f'{base_fg}_{base_bg}.flac'
            if out_name in used_names:
                k = 2
                while f'{base_fg}_{base_bg}__{k}.flac' in used_names:
                    k += 1
                out_name = f'{base_fg}_{base_bg}__{k}.flac'
            used_names.add(out_name)
            out_path = os.path.join(args.output_audio_dir, out_name)

            if args.skip_existing and os.path.exists(out_path):
                skipped_audio += 1
            else:
                sf.write(out_path, mixed, args.target_sr, format='FLAC')

            entry = build_entry(
                fg_e, bg_e, COMBO_FULL[combo], out_path, args.target_sr,
                L_fg, L_bg_orig, repeat_count, fg_offset_sec, snr, total_len,
            )
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            written += 1
            combo_counts[combo] += 1
            cell_counts[(combo, fg_e['category'], bg_e['category'])] += 1

            if written % 500 == 0:
                print(f'  ... {written} written '
                      f'(combo={combo}, fg={fg_e["category"]}, bg={bg_e["category"]})')

    # ─────────── summary ───────────
    print(f'[done] total written: {written} → {args.output_jsonl}')
    print(f'       audio dir:     {args.output_audio_dir}')
    print('[combo]')
    for c in COMBO_FULL:
        print(f'  {COMBO_FULL[c]:<9} = {combo_counts[c]}')
    print('[cells] (combo × fg_category × bg_category)')
    for c in COMBO_FULL:
        for fg_cat in ('speech', 'singing'):
            row = ', '.join(
                f'{bg_cat}:{cell_counts[(c, fg_cat, bg_cat)]}'
                for bg_cat in ('music', 'sound'))
            print(f'  {COMBO_FULL[c]:<9}  fg={fg_cat:<8}  →  {row}')
    print('[fg leftover]')
    print(f'       V^r: used {u_vr}/{Vr} (leftover {Vr - u_vr})')
    print(f'       V^f: used {u_vf}/{Vf} (leftover {Vf - u_vf})')
    print('[bg leftover]')
    print(f'       B^r: used {u_br}/{Br} (leftover {Br - u_br})')
    print(f'       B^f: used {u_bf}/{Bf} (leftover {Bf - u_bf})')
    if args.skip_existing:
        print(f'       skipped existing audio: {skipped_audio}')
    print(f'       seed={args.seed} (+salt {args.seed_salt} → {effective_seed}), '
          f'snr=[{args.snr_min}, {args.snr_max}] dB, target_sr={args.target_sr}')


if __name__ == '__main__':
    main()
