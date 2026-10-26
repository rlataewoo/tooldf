"""
plugin_tooldf.py
================

External plugin for `swift rlhf --rlhf_type grpo` that provides
ToolDF-specific reward functions.

Wire-up example (script/train_grpo_tooldf.sh):

    swift rlhf \\
        --rlhf_type grpo \\
        --model <SFT-merged-ckpt> \\
        --dataset label/Tool-GRPO/<train>.jsonl \\
        --external_plugins plugin/plugin_tooldf.py \\
        --reward_funcs tooldf_segment tooldf_answer tooldf_tool_selection \\
                       tooldf_consistency tooldf_format \\
        --reward_weights 1.0 0.5 0.3 0.2 0.1

Dataset columns expected per sample:
    messages : standard swift conversation (with <audio> in user)
    solution : SFT-style assistant content (string).
               Must contain
                 <audio_understanding>[s(s)-e(s)] desc ... </audio_understanding>
                 <plan>...</plan>
                 <answer>real|fake</answer>
               and optionally <tool_call>{...}</tool_call> blocks.
               In practice, copy the assistant turn(s) from the corresponding
               Tool-SFT JSONL sample into one string.
    gt_segments (optional) : list[{"type","start","end"}].
               If provided, it overrides parsing the `solution` for segment GT.
               Use this when paper-grade GT segments are available.

Env knobs (all optional):
    TOOLDF_FRAME_HOP        frame resolution in seconds (default 0.2)
    TOOLDF_TOLERANCE_TAU    tolerance F1 minimum collar in seconds (default 0.2)
    TOOLDF_TOLERANCE_PCT    DCASE-style fractional collar of event length
                            (default 0.2 — i.e. effective collar per GT event
                            is max(TAU, PCT * gt_length)); set 0 to disable.
    TOOLDF_FRAME_WEIGHT     weight on frame_f1 vs tolerance_f1 (default 0.7)
    TOOLDF_CONSIST_TAU      AU-vs-tool_call consistency tolerance (default 0.1)
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

try:
    # swift >= 4.0: swift.plugin split into swift.rewards + swift.rollout
    from swift.rewards import ORM, orms
    from swift.rollout import multi_turns
except ImportError:
    from swift.plugin import ORM, orms, multi_turns
try:
    # swift >= 4.0
    from swift.rollout.multi_turn import MultiTurnScheduler
    _HAS_MT_SCHEDULER = True
except ImportError:
    try:
        # swift >= 3.12: class-based MultiTurnScheduler API
        from swift.plugin.multi_turn import MultiTurnScheduler
        _HAS_MT_SCHEDULER = True
    except ImportError:
        # swift 3.5.3 etc.: function-based callback API only
        MultiTurnScheduler = object  # type: ignore
        _HAS_MT_SCHEDULER = False
from swift.utils import get_logger

logger = get_logger()


# ============================================================================
# Constants
# ============================================================================

CONTENT_TYPES = ('speech', 'singing', 'music', 'sound')

# detector tool name -> the audio content type it judges
TOOL_TO_TYPE: Dict[str, str] = {
    'speech_deepfake_detector':  'speech',
    'singing_deepfake_detector': 'singing',
    'music_deepfake_detector':   'music',
    'sound_deepfake_detector':   'sound',
}

# ============================================================================
# Parsers
# ============================================================================

_ANSWER_RE = re.compile(r'<answer>\s*(real|fake)\s*</answer>', re.IGNORECASE)
_AU_RE = re.compile(r'<audio_understanding>(.*?)</audio_understanding>', re.DOTALL)
_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
_DESC_RE = re.compile(r'<description>(.*?)</description>', re.DOTALL)
_DESC_SEG_RE = re.compile(
    r'\[\s*\d+(?:\.\d+)?\s*\(s\)\s*-\s*\d+(?:\.\d+)?\s*\(s\)\s*\]'
    r'(.*?)(?=\[\s*\d+(?:\.\d+)?\s*\(s\)|\Z)',
    re.DOTALL,
)
_VERDICT_RE = re.compile(r'\b(real|fake)\b', re.IGNORECASE)

# [<start>(s)-<end>(s)] <description until next bracket or end>
_SEG_RE = re.compile(
    r'\[\s*(\d+(?:\.\d+)?)\s*\(s\)\s*-\s*(\d+(?:\.\d+)?)\s*\(s\)\s*\]'
    r'(.*?)(?=\[\s*\d+(?:\.\d+)?\s*\(s\)|\Z)',
    re.DOTALL,
)


def parse_answer(text: str) -> Optional[str]:
    """Return 'real' / 'fake' / None."""
    m = _ANSWER_RE.search(text)
    return m.group(1).lower() if m else None


def parse_tool_call_segments(text: str) -> List[Tuple[float, float, str]]:
    """Extract (start, end, type) from every <tool_call>{...}</tool_call> JSON block.

    Only detector tools (TOOL_TO_TYPE) contribute segments; source_separator is ignored.
    """
    out: List[Tuple[float, float, str]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get('name')
        t = TOOL_TO_TYPE.get(name)
        if t is None:
            continue
        args = obj.get('arguments') or {}
        s, e = args.get('start'), args.get('end')
        if not (isinstance(s, (int, float)) and isinstance(e, (int, float))):
            continue
        s, e = float(s), float(e)
        if e <= s:
            continue
        out.append((s, e, t))
    return out


def _segments_from_gt_field(gt_segments_field) -> List[Tuple[float, float, str]]:
    """Coerce the optional `gt_segments` dataset column into the standard tuple form."""
    out = []
    if not gt_segments_field:
        return out
    for d in gt_segments_field:
        try:
            t = d['type']
            s = float(d['start'])
            e = float(d['end'])
        except (KeyError, TypeError, ValueError):
            continue
        if t in CONTENT_TYPES and e > s:
            out.append((s, e, t))
    return out


def _assistant_text(completion: str, messages) -> str:
    """Reconstruct the *full* assistant rollout text across all turns.

    swift's GRPO trainer passes only `messages[-1]['content']` (the final
    assistant turn) as `completion`. In multi-turn ToolDF rollouts, the
    interesting tags (<audio_understanding>, <plan>, <tool_call>) live in the
    FIRST assistant turn, while <description>/<answer> live in the LAST one.
    Reward functions need both, so we splice every assistant message together
    when the full message history is available.
    """
    if not messages:
        return completion or ''
    parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get('role') != 'assistant':
            continue
        content = m.get('content') or ''
        if not content or content == '<None>':
            continue
        parts.append(content)
    if not parts:
        return completion or ''
    return '\n'.join(parts)


def _get_msg_col(messages, n: int):
    """Normalize the `messages` kwarg into a list of length n, one entry per generation."""
    if messages is None:
        return [None] * n
    # When swift batches via rows_to_batched, this is already a list of lists.
    if isinstance(messages, list) and (len(messages) == n or n == 0):
        return messages
    return [None] * n


# ============================================================================
# Scoring helpers
# ============================================================================

def _frame_labels(segments: List[Tuple[float, float, str]],
                  duration: float,
                  hop: float) -> List[set]:
    """Discretize timeline; each frame is a set of types active in that frame (multi-label)."""
    n = max(1, int(math.ceil(duration / hop)))
    frames: List[set] = [set() for _ in range(n)]
    for s, e, t in segments:
        i0 = max(0, int(math.floor(s / hop)))
        i1 = min(n, int(math.ceil(e / hop)))
        for i in range(i0, i1):
            frames[i].add(t)
    return frames


def frame_multilabel_f1(pred: List[Tuple[float, float, str]],
                        gt: List[Tuple[float, float, str]],
                        hop: float = 0.1) -> float:
    """Type-aware multi-label F1 over fixed-resolution frames.

    Robust to over/under-segmentation and naturally handles concurrent types
    (e.g. c3: speech + sound at the same time).
    """
    if not gt:
        return 0.0
    duration = max(
        ([e for _, e, _ in pred] or [0.0])
        + ([e for _, e, _ in gt] or [0.0])
        + [hop],  # guard: at least one frame
    )
    pf = _frame_labels(pred, duration, hop)
    gf = _frame_labels(gt, duration, hop)
    tp = fp = fn = 0
    for p, g in zip(pf, gf):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def tolerance_f1(pred: List[Tuple[float, float, str]],
                 gt: List[Tuple[float, float, str]],
                 tau: float = 0.2,
                 tau_pct: float = 0.2) -> float:
    """DCASE-style event-based F1.

    A pred matches a gt iff same type AND both boundaries within an effective
    collar of `max(tau, tau_pct * (ge - gs))`. This mirrors `sed_eval`'s
    `t_collar` + `percentage_of_length` (Mesaros et al. 2016): short events
    use the floor `tau`, long events get a proportional collar.

    Set tau_pct=0 to recover the legacy fixed-tau behaviour.
    """
    if not pred and not gt:
        return 0.0
    used = set()
    tp = 0
    for ps, pe, pt in pred:
        for j, (gs, ge, gtype) in enumerate(gt):
            if j in used or gtype != pt:
                continue
            collar = max(tau, tau_pct * (ge - gs)) if tau_pct > 0 else tau
            if abs(ps - gs) <= collar and abs(pe - ge) <= collar:
                used.add(j)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


# ============================================================================
# Reward functions (ORMs)
# ============================================================================

class ToolDFSegmentReward(ORM):
    """PRIMARY reward: how well does the model's audio_understanding segmentation
    match the GT (start, end, content_type) tuples — at sub-second resolution.

        score = w_frame * frame_f1(hop=0.1s) + (1 - w_frame) * tolerance_f1(tau=0.5s)

    Prediction source : <audio_understanding> first; <tool_call> as fallback.
    GT source         : `gt_segments` column first; else parse `solution` (AU then tool_call).
    """

    def __init__(self) -> None:
        self.hop      = float(os.getenv('TOOLDF_FRAME_HOP',     '0.2'))
        self.tau      = float(os.getenv('TOOLDF_TOLERANCE_TAU', '0.2'))
        self.tau_pct  = float(os.getenv('TOOLDF_TOLERANCE_PCT', '0.2'))
        self.frame_w  = float(os.getenv('TOOLDF_FRAME_WEIGHT',  '0.7'))
        self.tol_w    = 1.0 - self.frame_w
        logger.info(f'[ToolDFSegmentReward] hop={self.hop}s, '
                    f'tau={self.tau}s, tau_pct={self.tau_pct}, '
                    f'frame_w={self.frame_w}, tol_w={self.tol_w}')

    @staticmethod
    def _extract_pred(completion: str) -> List[Tuple[float, float, str]]:
        return parse_tool_call_segments(completion)

    @staticmethod
    def _extract_gt(solution: str, gt_segments_field) -> List[Tuple[float, float, str]]:
        gt = _segments_from_gt_field(gt_segments_field)
        if gt:
            return gt
        return parse_tool_call_segments(solution)

    def __call__(self, completions, solution=None, messages=None, **kwargs) -> List[float]:
        n = len(completions)
        sols = solution if solution is not None else [''] * n
        gt_field_col = kwargs.get('gt_segments') or [None] * n
        msg_col = _get_msg_col(messages, n)

        rewards: List[float] = []
        for comp, sol, gt_field, msgs in zip(completions, sols, gt_field_col, msg_col):
            full = _assistant_text(comp, msgs)
            pred = self._extract_pred(full)
            gt = self._extract_gt(sol or '', gt_field)
            if not gt:
                # No usable GT -> zero reward (and warn once is enough; logging per-sample is noisy)
                rewards.append(0.0)
                continue
            f1_frame = frame_multilabel_f1(pred, gt, hop=self.hop)
            f1_tol   = tolerance_f1(pred, gt, tau=self.tau, tau_pct=self.tau_pct)
            rewards.append(self.frame_w * f1_frame + self.tol_w * f1_tol)
        return rewards


orms['tooldf_segment'] = ToolDFSegmentReward


class ToolDFAUToolConsistency(ORM):
    """Sanity reward: do the segment brackets declared in <audio_understanding>
    match the timestamps actually passed to <tool_call> args?

    Compares (start, end) tuples only — segment type is intentionally NOT
    derived from the AU free-text (keyword heuristics misclassify whenever
    the template literally contains words like "speech" or "music"). Type
    alignment is enforced by the detector-tool selection reward instead.

    Catches the failure mode where AU lists plausible segments but the tool_call
    uses different start/end — i.e. the model "talks the talk" but doesn't act on it.

    Score = tolerance_f1(AU_timestamps, tool_call_timestamps, tau=0.1s).
    No GT needed; pure self-consistency.
    """

    def __init__(self) -> None:
        self.tau = float(os.getenv('TOOLDF_CONSIST_TAU', '0.1'))

    @staticmethod
    def _au_timestamps(text: str) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        for au_match in _AU_RE.finditer(text):
            body = au_match.group(1)
            for m in _SEG_RE.finditer(body):
                try:
                    s, e = float(m.group(1)), float(m.group(2))
                except ValueError:
                    continue
                if e > s:
                    out.append((s, e, '_'))
        return out

    def __call__(self, completions, messages=None, **kwargs) -> List[float]:
        msg_col = _get_msg_col(messages, len(completions))
        rewards = []
        for comp, msgs in zip(completions, msg_col):
            full = _assistant_text(comp, msgs)
            au = self._au_timestamps(full)
            tc = [(s, e, '_') for s, e, _t in parse_tool_call_segments(full)]
            if not au or not tc:
                rewards.append(0.0)
                continue
            rewards.append(tolerance_f1(au, tc, tau=self.tau, tau_pct=0.0))
        return rewards


orms['tooldf_consistency'] = ToolDFAUToolConsistency


class ToolDFAnswerReward(ORM):
    """Outcome reward: <answer>real|fake</answer> matches the GT answer in `solution`.

    Binary 1.0 / 0.0. If either side is unparsable, returns 0.0.
    """

    def __call__(self, completions, solution, messages=None, **kwargs) -> List[float]:
        msg_col = _get_msg_col(messages, len(completions))
        rewards = []
        for comp, sol, msgs in zip(completions, solution, msg_col):
            full = _assistant_text(comp, msgs)
            pred = parse_answer(full)
            gt   = parse_answer(sol or '')
            if pred is None or gt is None:
                rewards.append(0.0)
            else:
                rewards.append(1.0 if pred == gt else 0.0)
        return rewards


orms['tooldf_answer'] = ToolDFAnswerReward


class ToolDFFormatReward(ORM):
    """Structural format check (lenient on tool_response interleaving):

        - exactly one <audio_understanding>...</audio_understanding>
        - exactly one <plan>...</plan>
        - exactly one <description>...</description>
        - exactly one <answer>real|fake</answer>
        - ordering: audio_understanding < plan < description < answer

    Tool-call blocks may appear between </think> and <description> in any number;
    we don't constrain them here (tooldf_tool_selection covers that).
    """

    @staticmethod
    def _single_pair(text: str, tag: str) -> bool:
        return text.count(f'<{tag}>') == 1 and text.count(f'</{tag}>') == 1

    def __call__(self, completions, messages=None, **kwargs) -> List[float]:
        msg_col = _get_msg_col(messages, len(completions))
        rewards = []
        for comp, msgs in zip(completions, msg_col):
            c = _assistant_text(comp, msgs)
            ok = (
                self._single_pair(c, 'audio_understanding')
                and self._single_pair(c, 'plan')
                and self._single_pair(c, 'description')
                and _ANSWER_RE.search(c) is not None
            )
            if not ok:
                rewards.append(0.0)
                continue
            try:
                i_au   = c.index('<audio_understanding>')
                i_plan = c.index('<plan>')
                i_desc = c.index('<description>')
                i_ans  = c.index('<answer>')
                ordered = i_au < i_plan < i_desc < i_ans
            except ValueError:
                ordered = False
            rewards.append(1.0 if ordered else 0.0)
        return rewards


orms['tooldf_format'] = ToolDFFormatReward


class ToolDFToolSelectionReward(ORM):
    """Did the model call the *right set* of detectors, and did it correctly
    decide whether to call source_separator?

        score = 0.7 * detector_jaccard(predicted_multiset, gt_multiset)
              + 0.3 * (1 if separator-presence matches GT else 0)

    Detector multiset = sorted list of detector tool names (with multiplicity).
    Source-separator is checked separately because it's a binary gating decision.
    """

    @staticmethod
    def _split_calls(text: str) -> Tuple[List[str], bool]:
        detectors: List[str] = []
        used_separator = False
        for m in _TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get('name')
            if name == 'source_separator':
                used_separator = True
            elif name in TOOL_TO_TYPE:
                detectors.append(name)
        return sorted(detectors), used_separator

    def __call__(self, completions, solution, messages=None, **kwargs) -> List[float]:
        msg_col = _get_msg_col(messages, len(completions))
        rewards = []
        for comp, sol, msgs in zip(completions, solution, msg_col):
            full = _assistant_text(comp, msgs)
            p_dets, p_sep = self._split_calls(full)
            g_dets, g_sep = self._split_calls(sol or '')

            cp, cg = Counter(p_dets), Counter(g_dets)
            inter = sum((cp & cg).values())
            union = sum((cp | cg).values())
            det_score = (inter / union) if union > 0 else 0.0

            sep_score = 1.0 if p_sep == g_sep else 0.0
            rewards.append(0.7 * det_score + 0.3 * sep_score)
        return rewards


orms['tooldf_tool_selection'] = ToolDFToolSelectionReward


class ToolDFAnswerOR(ORM):
    """OR-aggregation self-consistency.

    Rule (per the system prompt the SFT used):
        any per-segment fake -> overall fake;  else real.

    We extract per-segment verdicts from <description> and OR them. The model's
    <answer> must match. No GT comparison — the actual real/fake of the audio
    is irrelevant; what matters is that the model faithfully aggregates whatever
    detector verdicts it received.

    Returns 0.0 if either <description> verdicts or <answer> are unparseable.
    """

    @staticmethod
    def _description_verdicts(text: str) -> List[str]:
        m = _DESC_RE.search(text or '')
        if not m:
            return []
        out: List[str] = []
        for seg in _DESC_SEG_RE.finditer(m.group(1)):
            v = _VERDICT_RE.search(seg.group(1))
            if v:
                out.append(v.group(1).lower())
        return out

    def __call__(self, completions, messages=None, **kwargs) -> List[float]:
        msg_col = _get_msg_col(messages, len(completions))
        rewards: List[float] = []
        for comp, msgs in zip(completions, msg_col):
            full = _assistant_text(comp, msgs)
            verdicts = self._description_verdicts(full)
            ans = parse_answer(full)
            if not verdicts or ans is None:
                rewards.append(0.0)
                continue
            expected = 'fake' if 'fake' in verdicts else 'real'
            rewards.append(1.0 if ans == expected else 0.0)
        return rewards


orms['tooldf_answer_or'] = ToolDFAnswerOR


class ToolDFDescriptionCoverage(ORM):
    """Did <description> mention every detector verdict the model actually received?

    Failure mode this catches: model calls N detectors (e.g. speech=fake,
    singing=real, sound=real), then writes a <description> that only summarizes
    the LAST tool result, drops the fake one, and outputs <answer>real</answer>.
    `tooldf_answer_or` can't see the omission because it only reads description
    verdicts; so we add an explicit coverage check.

    Score = 1.0 iff (# detector tool_calls) == (# segment blocks in <description>),
    else 0.0. source_separator is excluded — it's not a verdict-producing call.
    """

    @staticmethod
    def _detector_call_count(text: str) -> int:
        n = 0
        for m in _TOOL_CALL_RE.finditer(text):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if obj.get('name') in TOOL_TO_TYPE:
                n += 1
        return n

    @staticmethod
    def _description_segment_count(text: str) -> int:
        m = _DESC_RE.search(text or '')
        if not m:
            return 0
        return sum(1 for _ in _DESC_SEG_RE.finditer(m.group(1)))

    def __call__(self, completions, messages=None, **kwargs) -> List[float]:
        msg_col = _get_msg_col(messages, len(completions))
        rewards: List[float] = []
        for comp, msgs in zip(completions, msg_col):
            full = _assistant_text(comp, msgs)
            n_calls = self._detector_call_count(full)
            n_segs  = self._description_segment_count(full)
            if n_calls == 0 or n_segs == 0:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if n_calls == n_segs else 0.0)
        return rewards


orms['tooldf_desc_coverage'] = ToolDFDescriptionCoverage


# ============================================================================
# Rollout: stubbed multi-turn scheduler (no real tool servers needed)
# ============================================================================

class ToolDFStubScheduler(MultiTurnScheduler):
    """Multi-turn rollout with synthetic tool_response — no real detectors / demucs.

    Why this exists
    ---------------
    plugin_tooldf's reward functions only parse the model's *own* text output
    (tags, tool_call args, <answer>). They do NOT inspect tool_response content.
    So driving a real detector / demucs server for every rollout is wasted work
    during RL — we can fabricate tool responses and skip the inference cost.

    Stub policy (GT-independent, intentionally mixed)
    -------------------------------------------------
    Each detector call independently flips a fair coin:

        prediction='real',  score ~ U[0.5, 1.0)        with prob TOOLDF_STUB_FAKE_P
        prediction='fake',  score ~ U[0.0, 0.5)        otherwise

    Mixed verdicts (e.g. [fake, real, real]) force the model to OR-aggregate;
    `tooldf_answer_or` reward then checks that the final <answer> matches the OR.
    The actual real/fake of the audio is intentionally NOT considered — RL teaches
    the *aggregation* rule, not the audio judgment (the SFT already does that).

    `source_separator` returns the dummy paths the model itself proposed (or
    'vocals.wav' / 'background.wav'). No file is actually created.

    Knobs
    -----
        TOOLDF_STUB_FAKE_P    per-call probability that detector returns 'fake'
                              (default 0.5 — uniform mix)

    Termination
    -----------
    Stops once the model emits <answer>real|fake</answer>, or hits max_turns,
    or the engine reports finish_reason='length' (handled by parent).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fake_p = float(os.getenv('TOOLDF_STUB_FAKE_P', '0.5'))

    def _stub_response(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == 'source_separator':
            return {
                'model':           'source_separator',
                'vocals_path':     args.get('vocals_path',     'vocals.wav'),
                'background_path': args.get('background_path', 'background.wav'),
            }
        if name in TOOL_TO_TYPE:
            try:
                s = float(args.get('start') or 0.0)
                e = float(args.get('end')   or 0.0)
            except (TypeError, ValueError):
                s = e = 0.0
            if random.random() < self.fake_p:
                pred, score = 'fake', round(random.uniform(0.0, 0.5), 4)
            else:
                pred, score = 'real', round(random.uniform(0.5, 1.0), 4)
            return {
                'model':      name,
                'start':      s,
                'end':        e,
                'prediction': pred,
                'score':      score,
            }
        return {'model': name, 'status': 'unknown_tool'}

    def step(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ''
        appended = False
        for m in _TOOL_CALL_RE.finditer(completion):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get('name')
            args = obj.get('arguments') or {}
            resp = self._stub_response(name, args)
            infer_request.messages.append({
                'role':    'tool_response',
                'content': json.dumps(resp, ensure_ascii=False),
            })
            appended = True
        if appended:
            # See tooldf_stub_handler: workaround for swift's _get_finish_reason
            # mislabeling long-audio prompts as length-clipped.
            try:
                response_choice.finish_reason = 'stop'
            except Exception:
                pass
        return {'infer_request': infer_request}

    def check_finished(self, infer_request, response_choice, current_turn) -> bool:
        completion = response_choice.message.content or ''
        if _ANSWER_RE.search(completion):
            return True
        return super().check_finished(infer_request, response_choice, current_turn)


# --- registry: prefer class-based scheduler if swift supports it; else fall back
# to a function-based callback compatible with swift 3.5.3 (`--multi_turn_func`).

_FAKE_P = float(os.getenv('TOOLDF_STUB_FAKE_P', '0.5'))


def _stub_response_func(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == 'source_separator':
        return {
            'model':           'source_separator',
            'vocals_path':     args.get('vocals_path',     'vocals.wav'),
            'background_path': args.get('background_path', 'background.wav'),
        }
    if name in TOOL_TO_TYPE:
        try:
            s = float(args.get('start') or 0.0)
            e = float(args.get('end')   or 0.0)
        except (TypeError, ValueError):
            s = e = 0.0
        if random.random() < _FAKE_P:
            pred, score = 'fake', round(random.uniform(0.0, 0.5), 4)
        else:
            pred, score = 'real', round(random.uniform(0.5, 1.0), 4)
        return {'model': name, 'start': s, 'end': e,
                'prediction': pred, 'score': score}
    return {'model': name, 'status': 'unknown_tool'}


def tooldf_stub_handler(inputs):
    """swift 3.5.3-style multi_turn_func.

    `inputs` is a list of dicts; each has 'messages' (list, last entry is the
    fresh assistant turn), and we set 'finished' True/False per item.

    Behaviour mirrors ToolDFStubScheduler:
      - parse <tool_call> blocks from the new assistant content
      - append a synthetic 'tool_response' message per call
      - terminate when <answer> is present, or when no tool_call was found
        (otherwise we'd loop forever once the model stops calling tools).

    swift `infer_engine.py:_get_finish_reason` bug workaround
    --------------------------------------------------------
    PT engine sets finish_reason='length' iff `num_prompt_tokens >= max_tokens`
    (where max_tokens = generation_config.max_new_tokens). For Qwen2.5-Omni
    the audio encoder emits ~50 tokens/sec, so any audio longer than ~20 s
    blows past max_completion_length and EVERY naturally-finished turn gets
    mislabeled as length-clipped. The GRPO trainer then exits the multi-turn
    loop on `finish_reason == 'length'`, so turn 2 (description+answer) never
    runs. We patch around it by overriding finish_reason -> 'stop' whenever
    we successfully queued at least one tool_response (i.e. the assistant did
    finish a parseable <tool_call> block, which it couldn't have done if it
    were truly truncated mid-token).
    """
    for inp in inputs:
        messages = inp.get('messages') or []
        if not messages:
            inp['finished'] = True
            continue

        last = messages[-1]
        completion = last.get('content', '') or '' if last.get('role') == 'assistant' else ''

        if _ANSWER_RE.search(completion):
            inp['finished'] = True
            # Same workaround: a clean <answer> means the model finished, not truncated.
            inp['finish_reason'] = 'stop'
            continue

        appended = False
        for m in _TOOL_CALL_RE.finditer(completion):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            name = obj.get('name')
            args = obj.get('arguments') or {}
            resp = _stub_response_func(name, args)
            messages.append({
                'role':    'tool_response',
                'content': json.dumps(resp, ensure_ascii=False),
            })
            appended = True

        if appended:
            # Override swift's mislabeled 'length' so the trainer continues to turn 2.
            inp['finish_reason'] = 'stop'

        # If no tool_call AND no answer -> the model is wandering; cut it off.
        inp['finished'] = (not appended)
    return inputs


if _HAS_MT_SCHEDULER:
    multi_turns['tooldf_stub'] = ToolDFStubScheduler
else:
    multi_turns['tooldf_stub'] = tooldf_stub_handler
