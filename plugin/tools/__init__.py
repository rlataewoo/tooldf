"""HTTP client wrappers for ToolDF tool calls.

The eval/training loop drives a Qwen2.5-Omni-3B agent that emits tool calls
matching `TOOL_SCHEMAS` in `script/infer/agent.py`. Each tool here is a thin
HTTP client to a separately-launched FastAPI server:

    - source_separator              -> run_demucs_api.py       (DEMUCS_URL)
    - speech_deepfake_detector      -> run_aasist_api.py x4    (SPEECH_DETECTOR_URL)
    - singing_deepfake_detector                                 (SINGING_DETECTOR_URL)
    - music_deepfake_detector                                   (MUSIC_DETECTOR_URL)
    - sound_deepfake_detector                                   (SOUND_DETECTOR_URL)

Every tool returns a JSON *string* — the eval loop puts this string straight
into a `tool` message for the model, so the wire format here must match what
the agent was trained on (see `prompt/ToolDF.txt`).
"""

from plugin.tools.separator import source_separator
from plugin.tools.detectors import (
    speech_deepfake_detector,
    singing_deepfake_detector,
    music_deepfake_detector,
    sound_deepfake_detector,
)

TOOLS = {
    'source_separator':          source_separator,
    'speech_deepfake_detector':  speech_deepfake_detector,
    'singing_deepfake_detector': singing_deepfake_detector,
    'music_deepfake_detector':   music_deepfake_detector,
    'sound_deepfake_detector':   sound_deepfake_detector,
}

__all__ = ['TOOLS']
