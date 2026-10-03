# Sort Images with Ollama

Classify and sort images into folders — screenshots, bad quality, and okay —
using a model that runs entirely on your own machine.

Two lanes, chosen with `CLASSIFIER` in `.env`:

- **`chat`** (default) — an ordinary chat model replies with a comma separated
  list of categories. Works with any vision model (`gemma3:4b`, `llava`, …).
- **`clef`** — a [Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/)
  decision model answers a *typed schema* in a single pass via Ollama's
  `/v1/systemone` endpoint and returns calibrated probabilities instead of free
  text. Needs Ollama ≥ 0.35.1. `clef-flash` (9B, ~11 GB) fits a 12 GB GPU;
  `clef:27b` (~18 GB) needs 24 GB or will spill into system RAM.

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
| `BAD_CATEGORIES` | `screenshot,blurry,low resolution,low quality` | chat-lane keyword match |
| `SUPPORTED_EXTENSIONS` | `.jpg,.jpeg,.png,.bmp,.webp` | file filter |
| `CLEF_MAX_PX` | `1280` | downscale before sending to Clef |
| `BLUR_THRESHOLD` | `0.5` | blur probability above which an image is junk |

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

Two caveats worth stating plainly:

- The answer key is a heuristic (or your own filing), **not ground truth**. The
  numbers measure *disagreement*, not who is right. Read the FP/FN lists.
- If the model only ever agrees with the heuristic, it is not earning its
  latency — keep the cheap path.

## Customization

- Edit `CLEF_QUESTIONS` in `sort-images.py` to change the schema Clef answers,
  and `CLEF_MOVE_MAP` to change where each answer goes.
- Edit the prompt in `classify_image()` for the chat lane.
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

## Contributing

Issues and pull requests are welcome.

## License

MIT — see [LICENSE](LICENSE).
