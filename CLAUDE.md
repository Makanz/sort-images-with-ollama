# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-user Python script that sorts images into `screenshots/`, `bad_quality/` and `ok/`
using a vision model served locally by Ollama. No cloud API, no dataset, no packaging — you run
`python sort-images.py` against a folder of images.

There are **two classifier lanes**, selected by `CLASSIFIER` in `.env`, and both must keep
working:

- **`chat`** (default) — `client.chat()` with a vision model; the model replies with a comma
  separated list of categories and the script substring-matches it against `BAD_CATEGORIES`.
  Works with any model, but the answer shape is not guaranteed. This is the fallback for people
  without a new enough Ollama or a big enough GPU.
- **`clef`** — `POST /v1/systemone` with a typed question schema. A *decision model* does one
  non-autoregressive forward pass and returns calibrated probabilities per option. Requires
  Ollama ≥ 0.35.1 and `ollama pull clef-flash`.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env
ollama serve                     # needs >= 0.35.1 for the clef lane

python sort-images.py            # the sorter; both lanes
python clef_bench.py <folder> --smoke    # one image, raw JSON — the first thing to try after pulling a model
python clef_bench.py "C:\photos\sample" --facit-mode folders --limit 200 --resume

python explain_sort.py --resume  # re-ask the model about images already in screenshots/ and bad_quality/
python explain_sort.py --smoke   # one image, raw answer, no files written
```

There is no test runner and no pytest. The tests are plain scripts under `tests/`, all runnable
from the repo root with no Ollama and no real images:

```bash
python tests/test_routing.py    # folder-routing decision table, both lanes
python tests/test_sort_run.py   # sort_images() end to end against a stub client, WORKERS=1 and 4
python tests/test_explain.py    # explain_sort.py: rule choice, mismatch kinds, HTML, --resume
```

Run all three after any change to routing, moving or the worker pool.

## Architecture

`sort-images.py` is the whole program. Import order is load-bearing:

1. Module-level config is read from the environment **at import time** (`os.getenv` calls at the
   top), and so is the `CLASSIFIER` validation (`sys.exit`), the `Client(host=...)`, and the
   `os.makedirs` of the three destination folders. Tests therefore set env vars *before*
   `exec_module`, and the destination folders are created simply by importing the module.
2. `classify_image()` (chat) / `classify_image_clef()` (clef) return a category set —
   `classify_image_clef` also returns Clef's `confidence`.
3. `resolve()` applies the optional second-opinion lane and returns `(categories, lane)`; the
   lane string is only for the log (`confidence=` / `spår=` in the output) so you can see
   afterwards how many images actually escalated.
4. `pick_folder()` maps a category set to a destination folder.
5. `sort_images()` lists `INPUT_FOLDER`, classifies in a `ThreadPoolExecutor(max_workers=WORKERS)`,
   and **moves each image in the main thread as soon as it finishes** (`as_completed`).

`clef_bench.py` is a separate, self-contained benchmark that measures either lane against an
answer key and reports TP/FP/FN/TN, precision/recall/F1 and latency, writing
`clef_bench_results.jsonl` (append-per-row, `--resume`) and `clef_bench_summary.md`.

`explain_sort.py` answers "why was this sorted out?" after the fact. The sorter persisted nothing
— every reason was a `print()` — so this re-asks the model about each image already sitting in the
destination folders and writes a self-contained HTML page (thumbnails as base64 JPEG data URIs;
browsers cannot render HEIC, which is why they are re-encoded) plus a `*_explain_results.jsonl`
cache. Two things to know before changing it:

- It is a **reader**: it never moves, renames or deletes anything in the image tree.
- Consequently **`--workers` is unconditionally safe here**, unlike in the sorter where only the
  main thread may move. There is no atomicity constraint; both the model and thumbnail passes use
  a pool, and only the JSONL append stays on the main thread.
- It copies `sort-images.py`'s `CLEF_QUESTIONS` (with `low_resolution`), **not** `clef_bench.py`'s
  `QUESTIONS`, so that "would this route differently" reproduces what the sorter would do. It
  compares **semantic buckets** (`screenshots`/`bad_quality`/`ok`), never raw folder basenames,
  because `CLEF_MOVE_MAP` hardcodes `"screenshots"` regardless of `SCREENSHOT_FOLDER_NAME`.
- Its JSONL stores the **full probability distribution**, and the verdict is derived at render
  time, so changing `--blur-threshold` re-derives the report without re-asking the model.
- A mismatch means the model **disagrees now**, not that the file is wrong — the chat lane is
  nondeterministic and a model change is indistinguishable from a real mis-sort. Never present the
  counts as ground truth.

## Invariants — do not break these

- **Moving, never deleting.** `get_unique_path()` must stay in the write path so an existing file
  is never overwritten. The existence check plus `shutil.move` is **not atomic**, so only the
  main thread may run them: classification is parallel, moving is not. Move as each image
  finishes — never in one batch at the end — so an interrupted run keeps the progress it made.
  `test_sort_run.py` asserts both properties (identical results at `WORKERS=1`/`4`, and that a
  move is visible while later classifications are still running).
- **The clef lane must use raw HTTP (`requests`), not `ollama.Client.systemone()`.** As of
  `ollama` 0.6.3 the client method has no `images` parameter, so it cannot do the one thing this
  lane exists for. Re-check before switching.
- **Clef accepts PNG/JPEG/WebP only, max 4 MiB and 16 Mpx per image.** Always downscale first
  (`CLEF_MAX_PX`); never send HEIC or RAW. Both lanes go through `requests`-level bodies that
  must stay byte-compatible with the endpoint.
- **Normalise category names before comparing across lanes.** Clef writes `low_resolution`, the
  chat lane's `BAD_CATEGORIES` free text says `low resolution`. Compare against the
  space→underscore normalised forms (`BAD_NORMALIZED`, `SCREENSHOT_NORMALIZED`, `CLEF_BAD`,
  `CLEF_MOVE_MAP`). The second model is asked in its own words, so its answer is normalised too.
- **The chat lane's vocabulary is closed, and every label it offers must be able to route.**
  `CHAT_PROMPT` offers exactly `BAD_NORMALIZED | SCREENSHOT_NORMALIZED` plus `ok`, and
  `parse_categories()` accepts only words in `CHAT_ALIASES`. This is not cosmetic: with an open
  prompt the model answered with *descriptions* (`wildlife trap, animal trap, outdoor, rodent`),
  and measured on 35 real images **16 (46 %) contained no keyword at all** and fell through to
  `ok/`. Adding a label to the prompt that the routing lists do not contain recreates that bug —
  `tests/test_routing.py` asserts the subset relation, so add it to `BAD_CATEGORIES` /
  `SCREENSHOT_CATEGORIES` first. Do not add substring matching to `parse_categories()`: it lets a
  stray `screenshot` inside a description through again, which is what put photos in
  `screenshots/`.
- **`CHAT_PROMPT` and `CHAT_ALIASES` are duplicated in `explain_sort.py`**, which cannot import the
  sorter and must ask the model the same question or the report stops measuring it. `tests/test_routing.py`
  pins both strings (and `CLEF_QUESTIONS`) against each other; `clef_bench.py` has its own stale copy
  that the guard does not cover.
- **Screenshot categories are checked before bad categories** in `pick_folder()`, because
  `screenshot` appears in both `SCREENSHOT_CATEGORIES` and `BAD_CATEGORIES`. Both lanes must
  produce the same three folders.
- **No heuristic libraries in the sorter.** Do not add `opencv`, `imagehash` or similar — the
  point of this repo is the model lane. Deterministic heuristics live in a separate project.
- **The second-opinion lane exists to be measured, not assumed.** All three modes (`veto`,
  `agree`, `override`) are computed from a single bench run on purpose; do not hardcode a winner.
- **`clef_bench.py`'s answer key is a heuristic, not ground truth.** Its numbers measure
  agreement with the key, never "accuracy".

## Conventions

- **Comments and docstrings are in Swedish; the README is in English.** Runtime output is also
  Swedish, with emoji prefixes (`📸 Classifying`, `✅ Sorted N of M`). Match the surrounding
  language — do not translate existing Swedish comments to English.
- The filename `sort-images.py` contains a hyphen, so it cannot be `import`ed normally. Tests load
  it with `importlib.util.spec_from_file_location`, give the module a unique name per load, set
  env vars first, then rebind the module global with `setattr(module, "client", ...)` to stub the
  model. Follow that pattern for new tests. Tests live in `tests/` and resolve the sorter — which
  sits a level up — with `Path(__file__).resolve().parents[1] / "sort-images.py"`.
- The chat lane hands the *path* (`ollama._types.Image(value=Path(...))`) to the model and never
  opens the file — which is why stub tests can use empty files. The clef lane must open and
  re-encode the image, and does need `pillow` + `pillow-heif` for HEIC.

## Gotchas

- The defaults in the source differ from the documented ones: `CLEF_MAX_PX` is `1280` and
  `BLUR_THRESHOLD` is `0.5` in code, while `.env.example` (and the README table) set `768` and
  `0.9` with the measurements that justify them. Treat `.env.example` as the intended config; the
  in-code values are only fallbacks.
- `WORKERS > 1` only pays off if the *model server* sets `OLLAMA_NUM_PARALLEL >= WORKERS` and
  restarts, **and** the model leaves VRAM over for a second slot (`clef-flash` ~11 GB does not on
  a 12 GB card). Otherwise the requests just queue and `WORKERS` adds threads and nothing else.
  Measure by running the same folder twice and watching `ollama ps` for `PROCESSOR` dropping below
  `100% GPU`.
- From inside the devcontainer, `localhost` is the container: use `host.docker.internal` or the
  host's LAN IP in `OLLAMA_HOST`.
- README's "Known blocker" section records that earlier claims about `clef-flash` being broken on
  Windows, and about Clef being text-only, were **disproven** on Ollama 0.40.2. It reports
  `capabilities: ['decision','vision']` and accepts images. Prefer `clef-flash` over `clef:27b`
  (same answers, 2.7 s vs 4.7 s median, fits 12 GB). A `500` on `/v1/systemone` should be verified
  with `--smoke` before anything else.
- `.env` is gitignored and may hold a different `SECOND_OPINION` than the example (the example
  ships it enabled).
