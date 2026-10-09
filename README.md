# Sort Images with Ollama

Classify and sort images into folders — screenshots, bad quality, and okay —
using a model that runs entirely on your own machine.

Two lanes, chosen with `CLASSIFIER` in `.env`:

- **`chat`** (default) — an ordinary chat model replies with a comma separated
  list of categories. Works with any vision model (`gemma3:4b`, `llava`, …).
  Screenshots land in their own folder here too, whenever the model names
  `screenshot` (or another word from `SCREENSHOT_CATEGORIES`).
- **`clef`** — a [Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/)
  decision model answers a *typed schema* in a single pass via Ollama's
  `/v1/systemone` endpoint and returns calibrated probabilities instead of free
  text. Needs Ollama ≥ 0.35.1. `clef-flash` (9B, ~11 GB) fits a 12 GB GPU and is the recommended
  variant; `clef:27b` (~18 GB) needs 24 GB or will spill into system RAM.

`clef_bench.py` measures either lane against a known-good answer key, so you can
see whether a model actually beats a simple heuristic before trusting it.

## Project structure

```
sort-images-with-ollama
├── .devcontainer/
├── .env.example
├── clef_bench.py          # benchmark Clef/Clef-flash against a facit
├── sort-images.py         # the sorter (both lanes)
└── README.md
```

## Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/Makanz/sort-images-with-ollama.git
   cd sort-images-with-ollama
   pip install -r requirements.txt
   ```

2. **Start Ollama and pull a model**
   ```bash
   ollama serve                 # needs >= 0.35.1 for the clef lane
   ollama pull gemma3:4b        # chat lane
   ollama pull clef-flash       # clef lane
   ```

3. **Configure**
   ```bash
   cp .env.example .env
   ```

### Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLASSIFIER` | `chat` | `chat` or `clef` |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama API URL |
| `MODEL_NAME` | `gemma3:4b` | model for the chat lane |
| `CLEF_MODEL_NAME` | `clef-flash` | `clef-flash` or `clef:27b` |
| `INPUT_FOLDER` | `images` | folder to sort |
| `BAD_QUALITY_FOLDER_NAME` | `bad_quality` | destination for junk |
| `OK_QUALITY_FOLDER_NAME` | `ok` | destination for keepers |
| `SCREENSHOT_FOLDER_NAME` | `screenshots` | destination for screenshots, both lanes |
| `BAD_CATEGORIES` | `screenshot,blurry,low resolution,low quality` | chat-lane keyword match |
| `SCREENSHOT_CATEGORIES` | `screenshot,photo of screen` | chat-lane keywords routed to `screenshots` instead of `bad_quality` |
| `SUPPORTED_EXTENSIONS` | `.jpg,.jpeg,.png,.bmp,.webp,.heic,.heif` | file filter |
| `CLEF_MAX_PX` | `768` | downscale before sending to Clef (768 is the measured sweet spot) |
| `BLUR_THRESHOLD` | `0.9` | blur probability above which an image is junk (0.9 = 0 false positives measured) |
| `SECOND_OPINION` | *(empty)* | set to `clef` to enable the second-opinion lane |
| `SECOND_OPINION_MODEL` | `gemma3:4b` | any vision chat model (`gemma3:4b`, `moondream`, `llava:7b`) |
| `SECOND_OPINION_THRESHOLD` | `0.8` | escalate when Clef's confidence falls below this |
| `SECOND_OPINION_MODE` | `veto` | `veto`, `agree` or `override` — see below |

## Second opinion

Clef already tells you how sure it is: every answer carries a `confidence` from
0 to 1. It measures how concentrated the probability mass is, not the chance the
answer is right — but low values reliably mean the model was torn between
options.

Set `SECOND_OPINION=clef` and, whenever that confidence drops below
`SECOND_OPINION_THRESHOLD`, a *different* model is asked about the same image.
Three ways to combine the two verdicts:

- **`veto`** (default, safest) — the second model can only *rescue* images. If
  Clef saw nothing wrong but the other model did, the image is quarantined
  anyway. Clef's own verdict always stands. This maximises recall: fewer bad
  images slip through.
- **`agree`** — quarantine only when *both* models flag a problem. Disagreement
  keeps the image. Most conservative: fewest false alarms, most junk retained.
- **`override`** — the second model's verdict replaces Clef's entirely.

Cost: the second model only runs on the images that actually escalated, so a
confident run pays nothing. Watch the `confidence=` and `spår=` values in the
output to see how often that happens; if almost everything escalates, raise
`SECOND_OPINION_THRESHOLD` or the second opinion is just doubling your runtime.

`clef_bench.py` computes **all three** modes from a single run, so you can
compare them without re-running anything.

## Usage

1. Put images in the `images` folder (or set `INPUT_FOLDER`).
2. Run:
   ```bash
   python sort-images.py
   ```
3. Files land in:
   - `images/screenshots/` — screenshots and photos of screens
   - `images/bad_quality/` — blurry, low resolution, documents
   - `images/ok/` — keepers

Both lanes produce all three folders. In the chat lane a screenshot goes to
`screenshots/` instead of `bad_quality/`, so junk screenshots can be swept
without touching genuinely blurry photos.

Files already in a destination folder, and non-images, are left alone. An
existing file is never overwritten — a numeric suffix is added instead.

## Benchmarking (`clef_bench.py`)

The interesting question isn't "does the model sound right" but "does it beat
what I already have". `clef_bench.py` answers that numerically: it asks Clef
about every image and compares the answers against an answer key, reporting
TP/FP/FN/TN, precision, recall, F1, agreement, and latency — plus lists of every
disagreement so you can eyeball them.

```bash
# Answer key from folder names: screenshots/  blurry/  ok/
python clef_bench.py "C:\photos\sample" --facit-mode folders --limit 200 --resume

# Answer key from a fotorens report (_Rensat_rapport.json, auto-detected)
python clef_bench.py "C:\photos" --limit 200 --resume
```

Output is `clef_bench_results.jsonl` (one row per image, resumable) and
`clef_bench_summary.md`. A `--smoke` flag sends one image and prints the raw
answer, which is the right first step after pulling the model.

With `--second-model gemma3:4b` the bench also escalates low-confidence images
and reports all three `--second-mode` variants side by side, plus how often the
two models disagreed. Same answer key, so the comparison is apples to apples:

```bash
python clef_bench.py "C:\photos\sample" --facit-mode folders \
  --second-model gemma3:4b --second-threshold 0.8
```

Two caveats worth stating plainly:

- The answer key is a heuristic (or your own filing), **not ground truth**. The
  numbers measure *disagreement*, not who is right. Read the FP/FN lists.
- If the model only ever agrees with the heuristic, it is not earning its
  latency — keep the cheap path.

## Customization

- Edit `CLEF_QUESTIONS` in `sort-images.py` to change the schema Clef answers,
  and `CLEF_MOVE_MAP` to change where each answer goes.
- Edit the prompt in `classify_image()` for the chat lane.
- Edit `SCREENSHOT_CATEGORIES` to change which chat-lane words route to
  `screenshots/`, and `SCREENSHOT_FOLDER_NAME` to rename the folder.
- Adjust `BLUR_THRESHOLD` and `CLEF_MAX_PX` to trade accuracy for speed.

## Troubleshooting

- **`Cannot connect to the LLM`** — check `OLLAMA_HOST`. From inside a
  container, `localhost` is the container; use `host.docker.internal` or the
  host's LAN IP.
- **`404 /v1/systemone`** — your Ollama is older than 0.35.1, or `CLASSIFIER` is
  `clef` while the model name isn't a decision model. Update Ollama.
- **Out of memory on the clef lane** — `clef:27b` needs ~18 GB. Use
  `clef-flash`, or lower `CLEF_MAX_PX`.
- **Every image lands in `ok`** — the model is answering `photo` for
  everything; run `clef_bench.py` to see whether that's actually correct on
  your images before assuming the sorter is broken.
- **HEIC/HEIF files are skipped or fail with `UnidentifiedImageError`** —
  `pillow-heif` isn't installed. It is in `requirements.txt`; confirm with
  `pip install pillow-heif`. Pillow cannot read HEIC on its own.

- **Second opinion always runs on every image** — Clef's `confidence` is below
  `SECOND_OPINION_THRESHOLD` for most of your images. Either the threshold is
  too high, or the task genuinely is hard and you are now paying for two models.
  Compare the modes in `clef_bench.py` output before committing to it.

## Contributing

Issues and pull requests are welcome.

## License

MIT — see [LICENSE](LICENSE).

## Known blocker: clef-flash on Windows — RESOLVED, and the earlier analysis was wrong

**Update 2026-10-09 (Ollama 0.40.2, RTX 3060 12 GB, Windows): `clef-flash` works.**
A live probe of 28 images with an exact answer key scored **28/28 correct image
type** (photo / screenshot / photo-of-screen), **0 false positives**, and **8/8
blurry photos** at `BLUR_THRESHOLD=0.9`. Every earlier claim below about
`clef-flash` being broken on Windows, and about Clef being text-only, is
disproven on this version.

Two things to take from that:

- **A capability string is a claim; an image POST is a finding.** `clef-flash`
  and `clef:27b` both report `capabilities: ['decision','vision']` and both
  accept `images`. `tev1` reports `['decision']` and rejects images with
  `400 "image inputs are not supported by this decision model"` — that is the
  text-only one.
- **A bug report rots.** Verify with the exact failing call before repeating a
  blocker to a user.

Prefer `clef-flash` over `clef:27b`: identical answers on the corpus, but 2.7 s
vs 4.7 s median per image, and it fits entirely in 12 GB VRAM while 27b spills
~10 GB to system RAM.

<details>
<summary>Superseded 2026-10-09 — the original (incorrect) blocker analysis</summary>

As of Ollama 0.35.1, `clef-flash` fails on **every** `/v1/systemone` request on
Windows with HTTP 500 `Clef: non-finite logit`. It is not a memory problem — the
model loads fully into VRAM — and it is not specific to your GPU: the same
failure occurs on CUDA, ROCm, Vulkan and CPU. The identical model blob works in
**WSL2 on the same GPU**, which points at the Windows build of Clef's decision
head rather than the model file. `clef:27b` (Q4_K_M) is reported working;
`clef-flash` (Q8_0) is the affected variant. `tev1` and `nimble` work fine.

Upstream: [ollama/ollama#18769](https://github.com/ollama/ollama/issues/18769).

Worth knowing before you build on this: **`clef` and `tev1` are text-only.**
Their capability is `decision`, not `vision` — they cannot score images. Only
`clef-flash` had a vision encoder, and it is the broken one. So on Windows today
there is no working image-capable decision model through `/v1/systemone`.
Options are WSL2, or a chat lane instead:

- `CLASSIFIER=chat` with a vision model — the original lane, unaffected.
- `CLASSIFIER=clef` against a WSL2 Ollama (`--host http://<wsl-ip>:11434`).

</details>

If you do ever hit the old 500, verify before debugging anything else:

```bash
python clef_bench.py <folder> --smoke    # a 500 here means an upstream regression
```
