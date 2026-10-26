# ToolDF: Tool-Integrated Reasoning for Mixed-Authenticity Audio Deepfake Detection

**Findings of EMNLP 2026**

![Paper](https://img.shields.io/badge/arXiv-2609.03620-b31b1b.svg)
![Checkpoints](https://img.shields.io/badge/Checkpoints-Google%20Drive-4285F4.svg)
![MAADD](https://img.shields.io/badge/MAADD-build%20locally-2ea44f.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

### Overview

Real-world manipulated audio can be **mixed-authenticity**: genuine and fake
components coexist across temporal transitions, overlapping sources, or both.
**ToolDF** uses an audio LLM (Qwen2.5-Omni-3B) as an orchestrator that analyzes the
audio scene, plans tool calls, runs source separation (Demucs) and domain-specific
detectors (XLSR-AASIST), and returns `fake` if any component is fake.

#### MAADD

| Subset | Setting | # Train | # Dev | # Eval |
| ------ | ------- | ------: | ----: | -----: |
| C1 | Speech ↔ Singing transition | 25,380 | 24,844 | 71,236 |
| C2 | Vocal + Music / Env. sound overlap | 109,783 | 46,130 | 47,039 |
| C3 | C1 + Music / Env. sound overlap | 25,380 | 18,024 | 12,084 |

MAADD audio is not redistributed; compose it locally (see [§2-2](#2-mixed-authenticity-add-dataset)).

### 1. Installation

```bash
# ToolDF & Demucs
conda create -n tooldf python==3.10.13 -y
conda activate tooldf
pip install -r requirements.txt
MAX_JOBS=4 pip install flash_attn --no-build-isolation 

conda create -n aasist python=3.10.13 -y
conda activate aasist
python -m pip install "pip<24.1" "setuptools<70" wheel
pip install -r requirements_aasist.txt
```

### 2. Data preparation

#### 1) Public Audio Deepfake Detection Datasets

- Our benchmark spans four audio domains. Please obtain the following datasets from their official sources:

| Domain        | Dataset               | Link                                                                                         |
| ------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| Speech        | ASVspoof 2019 (LA)    | [https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336) |
| Singing Voice | CtrSVDD (train & dev) | [https://zenodo.org/records/10467648](https://zenodo.org/records/10467648)                   |
| Singing Voice | CtrSVDD (eval)        | [https://zenodo.org/records/12703261](https://zenodo.org/records/12703261)                   |
| Sound         | ESDD (train & dev)    | [https://zenodo.org/records/16684355](https://zenodo.org/records/16684355)                   |
| Sound         | ESDD (eval)           | [https://zenodo.org/records/16685039](https://zenodo.org/records/16685039)                   |
| Music         | FakeMusicCaps (Fake)  | [https://zenodo.org/records/15063698](https://zenodo.org/records/15063698)                   |
| Music         | MusicCaps (Real)      | [https://www.kaggle.com/datasets/googleai/musiccaps](https://www.kaggle.com/datasets/googleai/musiccaps) |

After downloading, place each dataset under dataset/ in the project root, following the layout below. 

```
dataset
├── ASVspoof2019/DS/LA
│   ├── ASVspoof2019_LA_train
│   ├── ASVspoof2019_LA_dev
│   ├── ASVspoof2019_LA_eval
├── ctrsvdd
│   ├── train_set
│   ├── dev_set
│   ├── test_set
├── EnvSDD
│   ├── development
│   ├── test
├── FakeMusicCaps
    ├── audioldm2
    ├── ...
    ├── real (from MusicCaps)
```

#### 2) Mixed-Authenticity ADD Dataset

The data pipeline has four stages. Run every script from the project root.

> **Note.** We cannot redistribute the MAADD audio files, because the licenses of some
> of the source datasets do not permit it. We provide the composition scripts instead: run
> Stage 2 to compose the audio locally from the datasets you obtained in §2-1.
> Composition is seeded and deterministic, so running it on the downloaded Stage 1
> captions reproduces exactly the MAADD data used in the paper.
> Only the Stage 1 and Stage 3 outputs (JSONL metadata) are downloadable, and both
> reference audio paths under `dataset/MAADD/` — Stage 2 is required either way.

**Easy path.** Download the [Stage 1 captions](https://drive.google.com/drive/folders/1qYExBX15cT0ovKj6XK5zE3zVwH_er6ol?usp=drive_link), then run the rest in one command:

```
# unpack the download into label/audio_caption/, then:
bash script/data/run.sh               # stages 2 and 3, every split and category
bash script/data/run.sh --dry-run     # print the commands, run nothing
```

**Per-stage details** — only needed to run a stage by hand.

  


In the commands below, `<split>` is one of `train | dev | eval`; `<category>` is one of
`speech | singing | music | sound`.

**Stage 0 — Binary-SFT labels.** Real/fake labels paired with the audio paths of §2-1.

**Download: [binary-SFT labels](https://drive.google.com/drive/folders/1EiD9Jh2NbABiCZJQlR4lKyZcpVksOmiN?usp=drive_link)** → unpack into `label/binary_sft/`.

These files contain only paths and labels, no audio. Their paths are relative to the
project root, so they resolve against the layout above.

**Stage 1 — Audio captioning.** Captions every clip in the source datasets.

```
bash script/data/run_audiocaption.sh <split> <category>
```

Reads `label/binary_sft/*.jsonl` and writes `label/audio_caption/<dataset>_<split>.jsonl`.
Captions are generated with
[Qwen3-Omni-30B-A3B-Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner),
served behind an OpenAI-compatible endpoint (e.g. vLLM) — set `--api-url` inside
the script to your own server.
**Or skip it and download the caption files directly: [Stage 1 output](https://drive.google.com/drive/folders/1qYExBX15cT0ovKj6XK5zE3zVwH_er6ol?usp=drive_link).**

**Stage 2 — Composition.** Composes the captioned clips into the C1/C2/C3 subsets.

```bash
bash script/data/run_compose_c1.sh <split>   # speech + singing
bash script/data/run_compose_c2.sh <split>   # vocal over background
bash script/data/run_compose_c3.sh <split>   # C1 over background
```

Writes audio to `dataset/MAADD/c*/<split>/` and metadata to
`label/audio_caption/c*_<split>.jsonl`. Composition is deterministic: the same
seed and inputs always yield the same subsets.

**Stage 3 — ToolDF-SFT formatting.** Converts the metadata into Tool-SFT training samples.

```bash
bash script/data/run_prepare_sft_single.sh <split> <category>   # public datasets
bash script/data/run_prepare_sft_c1.sh <split>
bash script/data/run_prepare_sft_c2.sh <split>
bash script/data/run_prepare_sft_c3.sh <split>
```

Writes `label/tooldf_sft/*.jsonl`.
**Or skip it and download the training data directly: [Stage 3 output](https://drive.google.com/drive/folders/1pjy3l3_sRTLC-Gayfp9oxGOHXT9AMwa3?usp=sharing).**

Each stage consumes the previous stage's output, so run them in order. Stage 0 is
download-only; Stages 1 and 3 can be downloaded instead of run. `run.sh` also
takes `--stages`, `--splits`, and `--categories` to run a subset.



### 3. Pretrained Model

**Base model.** Only needed if you want to train (§4) or merge your own LoRA.

```bash
huggingface-cli download Qwen/Qwen2.5-Omni-3B --local-dir /mnt/Qwen2.5-Omni-3B
```

**ToolDF checkpoints.** To run eval (§5) you need our released weights: the merged
ToolDF-SFT agent checkpoint plus the XLSR-AASIST detectors and their XLSR-2 base.
Download the `pretrained/` archive and unpack it at the **project root** so the
paths below resolve as-is.

**Download: [pretrained/](https://drive.google.com/drive/folders/19S1NcfUmJZuUwkHq35GdrX76WMuUNLRM?usp=sharing)** → unpack into `pretrained/`.

```
pretrained
├── qwen2.5-omni-3b-tooldf-sft
│   └── v0-.../checkpoint-*-merged          # agent checkpoint  -> eval MODEL_PATH
├── xlsr2_300m.pt                            # shared XLSR-2 300M base
├── XLSR_AASIST_Speech_VL/best_model.pth     # general: raw input.wav  (ports 8021..8024)
├── XLSR_AASIST_Singing_VL/best_model.pth
├── XLSR_AASIST_Music_VL/best_model.pth
├── XLSR_AASIST_Sound_VL/best_model.pth
├── XLSR_AASIST_Speech_VL_Aug/best_model.pth # sep-aware: demucs stems (ports 8025..8028)
├── XLSR_AASIST_Singing_VL_Aug/best_model.pth
├── XLSR_AASIST_Music_VL_Aug/best_model.pth
└── XLSR_AASIST_Sound_VL_Aug/best_model.pth
```

The AASIST servers (§5-2) and the ToolDF eval (§5-3) read these paths by default;
override any of them via the env vars documented at the top of each script.

The XLSR-AASIST detectors (`xlsr2_300m.pt` + `XLSR_AASIST_*`) are trained with the
code from [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing),
one detector per audio domain (speech/singing/music/sound), in two variants:
the **general** detector scores the raw input audio, while the **separator-aware**
(`_Aug`) detector scores the demucs-separated stems (vocals/background) and is
trained with separation-artifact augmentation. Use our released weights above, or
retrain your own with that repo and drop the resulting `best_model.pth` into the
matching `XLSR_AASIST_*` folder.

### 4. Training ToolDF-SFT

LoRA-SFT the base Qwen2.5-Omni-3B into the ToolDF agent, then merge the adapter.
Runs in the `tooldf` conda env. Skip this and use a released checkpoint if you
only want to run eval.

**1) LoRA SFT.** Trains on all seven ToolDF-SFT train sets (`19la` / `ctrsvdd` /
`fakemusiccaps` / `ESDD` + `c1` / `c2` / `c3`), validating on the matching dev sets.

```bash
conda activate tooldf
bash script/train/train_tooldf_sft.sh
```

Edit the paths/knobs at the top of the script before running:
`--model` (base model from §3), `NPROC_PER_NODE` + `CUDA_VISIBLE_DEVICES`
(2 GPUs by default), `--system prompt/ToolDF.txt` and `--agent_template hermes`
(**must match eval**). Checkpoints land in
`output/qwen2.5-omni-3b-tooldf-sft/v<N>-<timestamp>/checkpoint-<step>`.

**2) Merge LoRA.** Fuses the adapter into the base weights for fast vLLM inference.

```bash
bash script/train/merge_lora.sh
```

Point `--adapters` at your trained checkpoint; swift writes a
`checkpoint-<step>-merged` directory next to it. Use that merged path as
`MODEL_PATH` in the eval scripts (§5).

### 5. Running ToolDF

ToolDF drives the agent through two tool servers (Demucs for source separation,
XLSR-AASIST for deepfake detection). Both must be up **before** the eval, so the
launch order is: **Demucs → AASIST → ToolDF eval**. The eval script health-checks
the two servers on startup and aborts if either is unreachable.

#### 1) Demucs source-separation server

Runs in the `demucs` conda env. The agent's `source_separator` tool calls it on
port `8001`.

For a **multi-GPU eval, Demucs must run multi-GPU too**: shard `i` calls Demucs on
port `8001 + i`, so an `N`-shard eval needs Demucs on ports `8001..8000+N`.
`--num-gpus N` launches exactly that — one server per GPU on consecutive ports.

```bash
conda activate demucs
pip install fastapi uvicorn

# single-GPU eval: one server on port 8001, GPU 0
bash script/infer/run_demucs_api.sh start

# multi-GPU eval: match --num-gpus to the number of eval shards (GPUs).
# This spans GPUs 0..7 and ports 8001..8008.
bash script/infer/run_demucs_api.sh start --num-gpus 8 --gpu 0

# stop (match the --port used at start; multi-GPU tears down all children)
bash script/infer/run_demucs_api.sh stop --port 8001
```

#### 2) XLSR-AASIST detector servers

Runs in the `aasist` conda env. Launches all eight detectors (general +
separator-aware × speech/singing/music/sound) on GPUs `0..7`, ports `8021..8028`.

```bash
conda activate aasist

# start all eight (override checkpoints/thresholds/ports/GPUs via env vars — see script header)
bash script/infer/run_aasist_api.sh start

# stop all
bash script/infer/run_aasist_api.sh stop
```

#### 3) ToolDF eval

Runs in the `tooldf` conda env. `<split>` selects the eval set
`label/tooldf_sft/<split>_eval_sft.jsonl` (name is case-sensitive):

| `<split>`       | Dataset                          |
| --------------- | -------------------------------- |
| `c1` `c2` `c3`  | MAADD mixed-authenticity subsets |
| `19la`          | ASVspoof 2019 LA (speech)        |
| `ctrsvdd`       | CtrSVDD (singing)                |
| `ESDD`          | ESDD (sound)                     |
| `fakemusiccaps` | FakeMusicCaps (music)            |

```bash
conda activate tooldf

# single GPU
bash script/infer/run_tooldf.sh <split>

# multi-GPU (data-parallel, sharded then merged).
# Requires Demucs started with --num-gpus = number of GPUs here (see step 1):
# shard i uses Demucs on port 8001+i.
GPUS="0 1 2 3 4 5 6 7" bash script/infer/run_tooldf_multigpu.sh <split>
```

Results are written to `result/tooldf_sft_<split>.jsonl`. Eval resumes
automatically from the last valid line, so re-running continues where it left off.

### 6. Metrics

```bash
python script/infer/eval_tooldf_metrics.py result/tooldf_sft_*.jsonl
```

### Citation

```bibtex
@inproceedings{kim-etal-2026-tooldf,
    title = "ToolDF: Tool-Integrated Reasoning for Mixed-Authenticity Audio Deepfake Detection",
    author = "Kim, Taewoo and Lee, Young Han and Park, Nam In and Kim, Chanwoo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2026",
    year = "2026"
}
```

### License

[MIT](LICENSE). Source datasets keep their own licenses.

