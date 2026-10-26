"""
SFT된 Qwen2.5-Omni ToolDF-SFT 체크포인트 추론 스크립트.

- agent_template='hermes' 로 tools를 system에 렌더링 + <tool_call> 파싱
- system prompt: prompt/ToolDF.txt (학습 시와 동일해야 함)
- Multi-turn 루프: 모델이 더 이상 tool_call을 내보내지 않을 때까지
  (tool_call → 실행 → 결과 주입)을 반복, 마지막에 <description>+<answer> 수집

사용법:
    python script/infer/agent.py \
        --model_path /mnt/NAS61/dataset_ssd/github/Qwen2.5-Omni-3B \
        --adapter_path output/ckpt_speech/qwen2.5-omni-3b-tool-sft/vX-XXX/checkpoint-XXX \
        --audio_path mix_sample.wav \
        --query "<audio>Is this audio real or fake?"
"""

import argparse
import json
import os
import re
import requests

import importlib.util as _ilu
# swift >= 4.0 flattened swift.llm into the top-level package and renamed
# PtEngine -> TransformersEngine. Decide by module presence, not try/except:
# importing these symbols can transitively pull in vllm._C, so a broken vLLM
# (e.g. missing libcudart) would otherwise masquerade as a swift-version error.
if _ilu.find_spec('swift.llm') is None:
    from swift import get_template, InferRequest, RequestConfig
    from swift import TransformersEngine as PtEngine
    from swift import get_processor
    get_model_tokenizer = None  # dropped in 4.x; see load_engine below
else:
    from swift.llm import get_model_tokenizer, get_template, PtEngine, InferRequest, RequestConfig
    get_processor = None
from swift.tuners import Swift

DEMUCS_URL = os.environ.get('DEMUCS_URL', 'http://localhost:8001')


# ─────────────────────────────────────────────
# Tool 실제 구현
# ─────────────────────────────────────────────
def source_separator(audio_path: str, vocals_path: str, background_path: str) -> str:
    """demucs 호출 후 모델에는 status만 노출 (경로 누출 방지)."""
    print(f'[tool] source_separator(audio_path={audio_path}, vocals_path={vocals_path}, background_path={background_path})')
    try:
        r = requests.post(
            f'{DEMUCS_URL}/separate',
            json={'audio_path': audio_path, 'vocals_path': vocals_path, 'background_path': background_path},
            timeout=600,
        )
        r.raise_for_status()
        status = r.json().get('status', 'completed')
        return json.dumps({'model': 'source_separator', 'status': status})
    except requests.RequestException as e:
        print(f'[tool] source_separator failed: {e}')
        return json.dumps({'model': 'source_separator', 'status': 'failed'})


def _detector_response(model_name, start, end, prediction, score):
    print(f'[tool] {model_name}(start={start}, end={end}) -> {prediction}@{score}')
    return json.dumps({
        'model': model_name,
        'start': start, 'end': end,
        'prediction': prediction, 'score': score,
    })


def speech_deepfake_detector(input_path, start, end, vocals_path=None):
    return _detector_response('speech_deepfake_detector', start, end, 'fake', 0.05)


def singing_deepfake_detector(input_path, start, end, vocals_path=None):
    return _detector_response('singing_deepfake_detector', start, end, 'real', 0.87)


def music_deepfake_detector(input_path, start, end, background_path=None):
    return _detector_response('music_deepfake_detector', start, end, 'real', 0.88)


def sound_deepfake_detector(input_path, start, end, background_path=None):
    return _detector_response('sound_deepfake_detector', start, end, 'real', 0.81)


TOOLS = {
    'source_separator': source_separator,
    'speech_deepfake_detector': speech_deepfake_detector,
    'singing_deepfake_detector': singing_deepfake_detector,
    'music_deepfake_detector': music_deepfake_detector,
    'sound_deepfake_detector': sound_deepfake_detector,
}


# ─────────────────────────────────────────────
# Tool 스키마 (학습 시와 바이트 단위 동일해야 함)
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


SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'prompt', 'ToolDF.txt')
with open(SYSTEM_PROMPT_PATH, 'r') as f:
    SYSTEM_PROMPT = f.read().strip()


# ─────────────────────────────────────────────
# Tool 실행
# ─────────────────────────────────────────────
def _coerce_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def execute_tool(name: str, args: dict, real_audio_path: str) -> str:
    if name not in TOOLS:
        return json.dumps({'error': f'unknown tool "{name}"'})
    # placeholder 강제 override
    if 'audio_path' in args:
        args['audio_path'] = real_audio_path
    if 'input_path' in args:
        args['input_path'] = real_audio_path
    try:
        return TOOLS[name](**args)
    except Exception as e:
        return json.dumps({'error': str(e)})


# ─────────────────────────────────────────────
# Fallback tool_call 파서 (hermes 파서가 실패했을 때 content 직접 추출)
# ─────────────────────────────────────────────
_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


class _FakeToolCall:
    """msg.tool_calls와 동일한 shape을 모사."""
    def __init__(self, name, arguments, idx):
        self.function = type('F', (), {'name': name, 'arguments': arguments})()
        self.id = f'toolcall-fallback-{idx}'
        self.type = 'function'


def _parse_tool_calls_from_content(content: str):
    out = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(content or '')):
        raw = m.group(1).strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get('name')
        args = obj.get('arguments', {})
        if not name:
            continue
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        out.append(_FakeToolCall(name, args, i))
    return out


# ─────────────────────────────────────────────
# Multi-turn 추론 루프
# ─────────────────────────────────────────────
def run_inference(engine, query, audios, max_tokens=1024, max_turns=8):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': query},
    ]
    real_path = audios[0] if audios else ''
    sep_vocals = sep_bg = None
    final_content = ''

    for turn in range(1, max_turns + 1):
        req = InferRequest(messages=messages, audios=audios, tools=TOOL_SCHEMAS)
        cfg = RequestConfig(max_tokens=max_tokens, temperature=0.0, stop=['</answer>'])
        resp = engine.infer([req], request_config=cfg)
        msg = resp[0].choices[0].message
        # print(msg)
        content = msg.content or ''
        tool_calls = getattr(msg, 'tool_calls', None) or []

        print('=' * 60)
        print(f'[Turn {turn} content]')
        print(content)

        if not tool_calls:
            final_content = content
            print('=' * 60)
            print('[Final answer]')
            print(final_content)
            return final_content

        print('=' * 60)
        print(f'[Turn {turn} parsed tool_calls: {len(tool_calls)}]')
        for tc in tool_calls:
            print(f'  - {tc.function.name}({tc.function.arguments})')

        # ── Tool 실행 (separator 후 분리 경로 자동 주입) ──
        tool_messages = []
        for i, tc in enumerate(tool_calls):
            name = tc.function.name
            args = _coerce_args(tc.function.arguments)

            if sep_vocals and name in ('speech_deepfake_detector', 'singing_deepfake_detector'):
                args['vocals_path'] = sep_vocals
            if sep_bg and name in ('music_deepfake_detector', 'sound_deepfake_detector'):
                args['background_path'] = sep_bg

            tc.function.arguments = json.dumps(args, ensure_ascii=False)
            result = execute_tool(name, args, real_path)

            if name == 'source_separator':
                try:
                    if json.loads(result).get('status') == 'completed':
                        sep_vocals = args.get('vocals_path')
                        sep_bg = args.get('background_path')
                except json.JSONDecodeError:
                    pass

            tool_messages.append({
                'role': 'tool',
                'content': result,
                'tool_call_id': getattr(tc, 'id', f'call_{turn}_{i}'),
            })

        print('=' * 60)
        print(f'[Turn {turn} tool results]')
        for tm in tool_messages:
            print(tm['content'])

        assistant_msg = {
            'role': 'assistant',
            'content': content,
            'tool_calls': [
                {
                    'id': getattr(tc, 'id', f'call_{turn}_{i}'),
                    'type': 'function',
                    'function': {
                        'name': tc.function.name,
                        'arguments': tc.function.arguments if isinstance(tc.function.arguments, str)
                                     else json.dumps(tc.function.arguments, ensure_ascii=False),
                    },
                }
                for i, tc in enumerate(tool_calls)
            ],
        }
        messages = messages + [assistant_msg] + tool_messages

    print('=' * 60)
    print(f'[Stopped: exceeded max_turns={max_turns}]')
    return final_content


# ─────────────────────────────────────────────
# Engine 로드
# ─────────────────────────────────────────────
def load_engine(model_path: str, adapter_path: str = None):
    print(f'[load] model: {model_path}')
    if adapter_path:
        print(f'[load] adapter: {adapter_path}')

    if get_model_tokenizer is None:
        # swift >= 4.0: get_template takes the processor first (template type
        # auto-detected), and TransformersEngine loads the model itself and
        # takes adapters directly -- there is no from_model_template.
        processor = get_processor(model_path)
        template = get_template(processor, agent_template='hermes')
        kw = {'adapters': [adapter_path]} if adapter_path else {}
        return PtEngine(model_path, template=template, **kw)

    model, tokenizer = get_model_tokenizer(model_path)
    if adapter_path:
        model = Swift.from_pretrained(model, adapter_path)
    template = get_template(
        model.model_meta.template,
        tokenizer,
        agent_template='hermes',
    )
    return PtEngine.from_model_template(model, template)


def main():
    parser = argparse.ArgumentParser(description='Tool-SFT Qwen2.5-Omni inference test')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--audio_path', type=str, default=None)
    parser.add_argument('--query', type=str, default='<audio>Is this audio real or fake?')
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--max_turns', type=int, default=8)
    args = parser.parse_args()

    engine = load_engine(args.model_path, args.adapter_path)
    audios = [args.audio_path] if args.audio_path else None
    run_inference(engine, args.query, audios, args.max_tokens, args.max_turns)


if __name__ == '__main__':
    main()
