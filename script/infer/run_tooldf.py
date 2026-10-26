"""
run_tooldf.py
=============

Run a ToolDF-SFT checkpoint over a ToolDF-SFT eval JSONL and dump per-sample
multi-turn results for downstream metrics (eval_metrics.py + ToolDF segment F1).

Output line schema (mirrors result/tooldf/tooldf_sft_*.jsonl, plus ToolDF fields):

    {
      "response":  <final assistant content (with <description>+<answer>)>,
      "labels":    <GT final assistant content>,
      "audios":    [...],
      "messages":  <original dataset messages>,
      "pred": {
          "answer":   "real|fake|null",
          "segments": [[start, end, type], ...],
          "full":     <concat of all assistant turns produced by the model>,
          "trace":    <full inference message list (system+user+assistant+tool)>
      },
      "gt": {
          "answer":   "real|fake|null",
          "segments": [[start, end, type], ...]
      }
    }

Tools are HTTP clients in plugin/tools/; servers are launched separately via
script/infer/run_demucs_api.sh and script/infer/run_aasist_api.sh. Per-sample
TemporaryDirectory makes vocals/background paths unique across concurrent
samples, regardless of what literal paths the agent emits.

Example:
    python script/infer/run_tooldf.py \
        --model_path /mnt/NAS61/dataset_ssd/github/Qwen2.5-Omni-3B \
        --adapter_path output/qwen2.5-omni-3b-tooldf-sft/v0-XXX/checkpoint-XXX \
        --dataset_path label/Tool-SFT/c3_eval_sft.jsonl
    # -> result/tooldf/tooldf_sft_c3.jsonl
"""

import argparse
import ctypes
import gc
import json
import os
import sys
import tempfile
import time
import traceback

# glibc holds freed memory in per-thread arenas instead of returning it to
# the OS, so RSS climbs even when Python objects are released. Call
# malloc_trim(0) periodically to actually shrink the heap.
try:
    _libc = ctypes.CDLL('libc.so.6')
    def _malloc_trim():
        _libc.malloc_trim(0)
except OSError:
    def _malloc_trim():
        pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, THIS_DIR)

# vLLM multi-modal input cache holds preprocessed audio embeddings keyed
# by input hash. Setting it to 0 breaks vLLM's profile_run (cachetools
# rejects any item larger than maxsize). Cap at 1 GiB instead — enough
# for the dummy mm item at startup, but small enough to bound RAM growth.
# Combined with the `audios=None for turn>0` patch in run_one, this caps
# at ~1 GiB across the whole run.
os.environ.setdefault('VLLM_MM_INPUT_CACHE_GIB', '1')
os.environ.setdefault('VLLM_MM_PROCESSOR_CACHE_GB', '1')

import importlib.util as _ilu

# swift >= 4.0 flattened swift.llm into the top-level package (VllmEngine,
# get_template, get_processor, InferRequest, ... all live on `swift`). Decide
# the version by module presence, NOT by try/except on a real import: importing
# swift.VllmEngine also pulls in vllm._C, so a broken vLLM install (e.g. a
# missing libcudart) would raise ImportError and get silently miscategorized as
# "wrong swift version". find_spec keeps the two failure modes distinct.
_SWIFT_V4 = _ilu.find_spec('swift.llm') is None

if _SWIFT_V4:
    from swift import InferRequest, RequestConfig
else:
    from swift.llm import InferRequest, RequestConfig

from agent import (
    TOOL_SCHEMAS,
    SYSTEM_PROMPT,
    _coerce_args,
    _parse_tool_calls_from_content,
    load_engine as load_pt_engine,
)
from plugin.plugin_tooldf import (
    parse_answer,
    parse_tool_call_segments,
)
from plugin.tools import TOOLS


def _build_template(model_path, is_v4):
    """Load the processor and build the hermes-agent template.

    swift 4.x: get_processor(path) + get_template(processor, ...)
    swift 3.x: get_model_tokenizer(path, load_model=False) + get_template(type, processor, ...)
    """
    if is_v4:
        from swift import get_processor, get_template
        processor = get_processor(model_path)
        return get_template(processor, agent_template='hermes')
    from swift.llm import get_model_tokenizer, get_template  # noqa: F811
    _, processor = get_model_tokenizer(model_path, load_model=False)
    return get_template(processor.model_meta.template, processor, agent_template='hermes')


def load_vllm_engine(model_path, adapter_path=None,
                     gpu_memory_utilization=0.85, max_model_len=8192,
                     enable_prefix_caching=True, enforce_eager=False,
                     limit_mm_per_prompt=None):
    """vLLM backend with prefix caching — multi-turn prefills become near-free
    when the prefix matches a previous turn."""
    if _SWIFT_V4:
        from swift import VllmEngine
    else:
        from swift.llm import VllmEngine
    print(f'[load:vllm] model: {model_path}')
    if adapter_path:
        print(f'[load:vllm] adapter (LoRA): {adapter_path}')
    template = _build_template(model_path, _SWIFT_V4)
    kw = dict(
        template=template,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=enforce_eager,
    )
    # swap_space (CPU buffer for KV offload) is a vLLM arg; the swift wrapper
    # exposes it differently across versions. Probe and pass via whichever
    # key this swift accepts. vLLM >= 0.10 (V1 engine) dropped swap_space
    # entirely -- there is no CPU KV swap to disable -- so only pass it when
    # vLLM's own EngineArgs still accepts it.
    vllm_params = _vllm_init_params()
    if _engine_args_accepts('swap_space'):
        if 'swap_space' in vllm_params:
            kw['swap_space'] = 0
        elif 'engine_kwargs' in vllm_params:
            kw['engine_kwargs'] = {'swap_space': 0}
    # LoRA adapters: ms-swift API differs across versions.
    #   - older swift: pass `adapters=[path]`
    #   - newer swift: pass `enable_lora=True` and register via LoRARequest
    # Easiest path: merge the LoRA into the base model first (script/merge_lora.sh)
    # and load the merged ckpt as model_path -- no adapter wiring needed.
    if adapter_path:
        if 'adapters' in _vllm_init_params():
            kw['adapters'] = [adapter_path]
        else:
            kw['enable_lora'] = True
            kw['max_loras'] = 1
            print('[load:vllm] WARNING: this swift VllmEngine does not accept '
                  '`adapters=`. Pass a *merged* checkpoint as --model_path '
                  'instead, or extend this loader to use LoRARequest.')
    if limit_mm_per_prompt:
        kw['limit_mm_per_prompt'] = limit_mm_per_prompt
    return VllmEngine(model_path, **kw)


def _vllm_init_params():
    import inspect
    if _SWIFT_V4:
        from swift import VllmEngine
    else:
        from swift.llm import VllmEngine
    return set(inspect.signature(VllmEngine.__init__).parameters)


def _engine_args_accepts(name):
    """Whether vLLM's own EngineArgs takes `name`. swift's VllmEngine forwards
    unknown kwargs straight through, so accepting a kwarg upstream says nothing
    about whether vLLM will."""
    import inspect
    from vllm import EngineArgs
    return name in inspect.signature(EngineArgs.__init__).parameters


def load_engine(engine_kind, model_path, adapter_path=None, **vllm_kwargs):
    if engine_kind == 'vllm':
        return load_vllm_engine(model_path, adapter_path, **vllm_kwargs)
    return load_pt_engine(model_path, adapter_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatch (real HTTP clients live in plugin/tools/).
#
# `audio_path` / `input_path` are always rewritten to the real eval audio —
# the agent emits literal "input_audio" placeholders from training. Separator
# write paths and detector vocals/background reads are pinned to a per-sample
# tmpdir by the caller (run_one) so concurrent samples never collide.
# ─────────────────────────────────────────────────────────────────────────────
def execute_tool(name, args, real_audio_path):
    if name not in TOOLS:
        return json.dumps({'error': f'unknown tool "{name}"'})
    if 'audio_path' in args:
        args['audio_path'] = real_audio_path
    if 'input_path' in args:
        args['input_path'] = real_audio_path
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return json.dumps({'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Single-sample multi-turn inference (silent).
# ─────────────────────────────────────────────────────────────────────────────
def run_one(engine, query, audios, max_tokens=1024, max_turns=8):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': query},
    ]
    real_path = audios[0] if audios else ''
    final_content = ''
    assistant_turns = []

    # Per-sample tmpdir guarantees vocals/background paths are unique even when
    # eval is sharded across processes — the agent always emits the literal
    # "vocals.wav" / "background.wav" from its training prompt, so we ignore
    # whatever paths the model produced and pin them here.
    with tempfile.TemporaryDirectory(prefix='tooldf_') as td:
        sep_vocals = os.path.join(td, 'vocals.wav')
        sep_bg     = os.path.join(td, 'background.wav')
        sep_done = False

        for turn in range(max_turns):
            # Only pass audios on turn 0: prefix caching reuses the audio
            # embedding for later turns, and re-passing it forces vLLM to
            # re-hash/re-cache the same waveform every turn (RAM growth).
            req = InferRequest(
                messages=messages,
                audios=audios if turn == 0 else None,
                tools=TOOL_SCHEMAS,
            )
            cfg = RequestConfig(max_tokens=max_tokens, temperature=0.0, stop=['</answer>'])
            resp = engine.infer([req], request_config=cfg)
            msg = resp[0].choices[0].message
            content = msg.content or ''
            tool_calls = getattr(msg, 'tool_calls', None) or []
            if not tool_calls:
                tool_calls = _parse_tool_calls_from_content(content)

            assistant_turns.append(content)

            if not tool_calls:
                final_content = content
                break

            tool_messages = []
            for i, tc in enumerate(tool_calls):
                name = tc.function.name
                args = _coerce_args(tc.function.arguments)

                # Force-unique paths regardless of what the model emitted.
                if name == 'source_separator':
                    args['vocals_path']     = sep_vocals
                    args['background_path'] = sep_bg
                if name in ('speech_deepfake_detector', 'singing_deepfake_detector'):
                    if sep_done and os.path.exists(sep_vocals):
                        args['vocals_path'] = sep_vocals
                if name in ('music_deepfake_detector', 'sound_deepfake_detector'):
                    if sep_done and os.path.exists(sep_bg):
                        args['background_path'] = sep_bg

                tc.function.arguments = json.dumps(args, ensure_ascii=False)
                result = execute_tool(name, args, real_path)

                if name == 'source_separator':
                    try:
                        if json.loads(result).get('status') == 'completed':
                            sep_done = True
                    except json.JSONDecodeError:
                        pass

                tool_messages.append({
                    'role': 'tool',
                    'content': result,
                    'tool_call_id': getattr(tc, 'id', f'call_{turn}_{i}'),
                })

            assistant_msg = {
                'role': 'assistant',
                'content': content,
                'tool_calls': [
                    {
                        'id': getattr(tc, 'id', f'call_{turn}_{i}'),
                        'type': 'function',
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                                if isinstance(tc.function.arguments, str)
                                else json.dumps(tc.function.arguments, ensure_ascii=False),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            }
            messages = messages + [assistant_msg] + tool_messages

    return {
        'final_content':   final_content,
        'assistant_turns': assistant_turns,
        'trace':           messages,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_eval_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def extract_user_query(messages, default='<audio>Is this audio real or fake?'):
    for m in messages:
        if m.get('role') == 'user':
            return m.get('content') or default
    return default


def gt_full_text(messages):
    """Concatenate the GT trace into a single string that mirrors how the
    model's own assistant output looks (with <tool_call>...</tool_call> blocks
    inlined). Used by segment parsers and <answer> parsers.

    The SFT data stores tool_call as its own role='tool_call' message (JSON
    only, no <tool_call> wrapper). We re-wrap it so parse_tool_call_segments
    can match it the same way it matches model-emitted text.
    """
    parts = []
    for m in messages:
        role = m.get('role')
        content = m.get('content', '') or ''
        if role == 'assistant':
            parts.append(content)
        elif role == 'tool_call':
            parts.append(f'<tool_call>\n{content}\n</tool_call>')
    return '\n'.join(parts)


def gt_final_assistant(messages):
    last = ''
    for m in messages:
        if m.get('role') == 'assistant':
            last = m.get('content', '') or last
    return last


def derive_output_path(dataset_path):
    base = os.path.basename(dataset_path)
    for suf in ('_eval_sft.jsonl', '_dev_sft.jsonl', '_train_sft.jsonl',
                '_eval.jsonl', '.jsonl'):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return os.path.join('result', 'tooldf', f'tooldf_sft_{base}.jsonl')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='Tool-SFT/GRPO eval runner')
    p.add_argument('--model_path',   required=True)
    p.add_argument('--adapter_path', default=None)
    p.add_argument('--dataset_path', required=True,
                   help='label/Tool-SFT/<name>_eval_sft.jsonl')
    p.add_argument('--output_path',  default=None,
                   help='Default: result/tooldf/tooldf_sft_<name>.jsonl')
    p.add_argument('--limit',        type=int, default=None,
                   help='Cap on samples (debug).')
    p.add_argument('--max_tokens',   type=int, default=1024)
    p.add_argument('--max_turns',    type=int, default=8)
    p.add_argument('--engine',       choices=['pt', 'vllm'], default='pt',
                   help='pt: PtEngine (eager). vllm: VllmEngine with prefix '
                        'caching (much faster for multi-turn).')
    # vLLM-only knobs
    p.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    p.add_argument('--max_model_len',          type=int,   default=8192)
    p.add_argument('--enforce_eager',          action='store_true',
                   help='vLLM: disable CUDA graphs (debug only).')
    p.add_argument('--no_prefix_caching',      action='store_true',
                   help='vLLM: turn off prefix caching (debug).')
    # data-parallel sharding
    p.add_argument('--num_shards', type=int, default=1,
                   help='Split the dataset into this many shards.')
    p.add_argument('--shard_id',   type=int, default=0,
                   help='Which shard this process handles (0..num_shards-1). '
                        'When num_shards>1, output_path gets a .shard{ID}.jsonl suffix.')
    # resume
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing output instead of resuming. '
                        'Default: if output exists, count valid JSON lines, '
                        'truncate any partial trailing line, and append.')
    # batch-restart: bound a single run so the caller can recycle the process
    # and release vLLM prefix-cache / MM-cache / glibc-arena accumulation.
    p.add_argument('--max_new', type=int, default=None,
                   help='Process at most this many NEW samples (after resume '
                        'skip) then exit. Resume on the next call picks up '
                        'where this run stopped.')
    args = p.parse_args()

    assert 0 <= args.shard_id < args.num_shards, \
        f'shard_id={args.shard_id} must be in [0, {args.num_shards})'

    # Pin demucs to the per-shard instance (port = DEMUCS_PORT_BASE +
    # shard_id). plugin.tools.separator reads this at call time. Honor any
    # pre-set value (e.g., from the launch script) so the caller wins.
    os.environ.setdefault('DEMUCS_SHARD_ID', str(args.shard_id))

    if args.output_path is None:
        args.output_path = derive_output_path(args.dataset_path)
    if args.num_shards > 1:
        root, ext = os.path.splitext(args.output_path)
        args.output_path = f'{root}.shard{args.shard_id}{ext}'
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    print(f'[evaluate_tooldf] dataset : {args.dataset_path}')
    print(f'[evaluate_tooldf] output  : {args.output_path}')
    if args.num_shards > 1:
        print(f'[evaluate_tooldf] shard   : {args.shard_id}/{args.num_shards}')

    rows = load_eval_jsonl(args.dataset_path)
    if args.num_shards > 1:
        rows = rows[args.shard_id::args.num_shards]
    if args.limit is not None:
        rows = rows[:args.limit]
    total = len(rows)
    print(f'[evaluate_tooldf] {total} samples')

    # Resume: count complete JSON lines in existing output, truncate any
    # partial trailing line, and skip those samples.
    done = 0
    if os.path.exists(args.output_path) and not args.overwrite:
        valid_bytes = 0
        with open(args.output_path, 'rb') as f:
            for line in f:
                try:
                    json.loads(line.decode('utf-8'))
                    done += 1
                    valid_bytes += len(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    break
        actual_size = os.path.getsize(args.output_path)
        if valid_bytes != actual_size:
            with open(args.output_path, 'r+b') as f:
                f.truncate(valid_bytes)
            print(f'[resume] truncated trailing partial bytes: '
                  f'{actual_size - valid_bytes}')
        if done >= total:
            print(f'[resume] already complete ({done}/{total}). Exiting without loading engine.')
            return
        if done > 0:
            print(f'[resume] skipping {done}/{total} already-done samples')
            rows = rows[done:]

    if args.max_new is not None:
        if args.max_new <= 0:
            print(f'[batch] max_new={args.max_new} <= 0; nothing to do.')
            return
        rows = rows[:args.max_new]
        print(f'[batch] this run will process at most {len(rows)} new samples')

    engine = load_engine(
        args.engine,
        args.model_path,
        args.adapter_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=not args.no_prefix_caching,
        enforce_eager=args.enforce_eager,
    )

    n = done
    t0 = time.time()
    mode = 'a' if done > 0 else 'w'
    with open(args.output_path, mode, encoding='utf-8') as fout:
        for row in rows:
            audios = row.get('audios') or []
            messages_in = row.get('messages') or []
            query = extract_user_query(messages_in)

            try:
                out = run_one(
                    engine, query, audios,
                    max_tokens=args.max_tokens,
                    max_turns=args.max_turns,
                )
                err = None
            except Exception as e:
                traceback.print_exc()
                out = {'final_content': '', 'assistant_turns': [], 'trace': []}
                err = str(e)

            full_pred = '\n'.join(out['assistant_turns'])
            gt_text = gt_full_text(messages_in)
            gt_last = gt_final_assistant(messages_in)

            pred_ans = parse_answer(full_pred)
            gt_ans   = parse_answer(gt_text)
            response_text = f'<answer>{pred_ans}</answer>' if pred_ans else ''
            labels_text   = f'<answer>{gt_ans}</answer>'   if gt_ans   else ''

            pred_segs = parse_tool_call_segments(full_pred)
            gt_segs   = parse_tool_call_segments(gt_text)

            rec = {
                'response': response_text,
                'labels':   labels_text,
                'audios':   audios,
                'messages': messages_in,
                'pred': {
                    'answer':   pred_ans,
                    'segments': [list(s) for s in pred_segs],
                    'full':     full_pred,
                    'trace':    out['trace'],
                },
                'gt': {
                    'answer':   gt_ans,
                    'segments': [list(s) for s in gt_segs],
                },
            }
            if err is not None:
                rec['error'] = err

            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()

            del out, rec
            n += 1
            if n % 50 == 0:
                gc.collect()
                _malloc_trim()
            if n % 10 == 0 or n == total:
                dt = time.time() - t0
                processed = n - done
                rate = processed / dt if dt > 0 else 0.0
                eta = (total - n) / rate if rate > 0 else float('inf')
                print(f'[evaluate_tooldf] {n}/{total} '
                      f'({rate:.2f} samp/s, eta {eta:.0f}s)')

    print(f'[evaluate_tooldf] done -> {args.output_path}')


if __name__ == '__main__':
    main()
