# Instructions for working in sort-images-with-ollama

## Overview

The project sorts images into `screenshots/`, `bad_quality/` and `ok/` using a
model that runs locally through Ollama. There is no cloud API and no dataset.

## Two classifier lanes

Everything hinges on `CLASSIFIER` in `.env`:

- **`chat`** — `client.chat()` with a vision model. The model returns free text
  and the script does a substring match against `BAD_CATEGORIES`. Cheap, any
  model, but the answer shape is not guaranteed. A word from
  `SCREENSHOT_CATEGORIES` routes the image to `screenshots/` before
  `BAD_CATEGORIES` gets a say, so both lanes produce the same three folders.
- **`clef`** — `POST /v1/systemone` with a typed question schema. The model
  returns calibrated probabilities per option. This is a *decision model*, not
  a chat model: it does one non-autoregressive forward pass. Requires Ollama
  ≥ 0.35.1.

Keep both lanes working. The chat lane is the fallback for people without a
new enough Ollama or a big enough GPU.

## Rules

- The clef lane must use **raw HTTP** (`requests`), not `ollama.Client.systemone()`.
  As of `ollama` 0.6.3 the client method has no `images` parameter, so it cannot
  do the one thing this lane exists for. Re-check before switching.
- Clef accepts PNG/JPEG/WebP only, max 4 MiB and 16 Mpx per image. Always
  downscale first (`CLEF_MAX_PX`), and never send HEIC or RAW.
- `ollama.systemone()` is the documented API name and `keep_alive` is accepted
  by the endpoint, but neither is guaranteed on older servers. Fail loudly.
- Sort by *moving*, never deleting. `get_unique_path()` must stay in the write
  path — never overwrite an existing file.
- Do not add a dependency on `opencv`, `imagehash` or any heuristic library to
  the sorter. The point of this repo is the model lane; deterministic heuristics
  live in a separate project.
- `clef_bench.py`'s answer key is a heuristic, not ground truth. Never present
  its numbers as accuracy; they measure agreement with the key.
- The second-opinion lane exists to be *measured*, not assumed. All three modes
  are computed from one run on purpose — do not hardcode a winner.
- The second model is asked in its own words (comma separated categories), not
  Clef's schema. Normalise its answer before comparing, or "low resolution"
  and "low_resolution" look like a disagreement.

## Testing

`python test_routing.py` checks the folder routing with no Ollama and no images
— it is the fast check for any change to `pick_folder()`.

Everything else is verified end-to-end against a local Ollama: put a
screenshot, a blurred photo and a sharp photo in `images/`, run the script,
confirm they land in different folders. For the clef lane, start with
`python clef_bench.py <folder> --smoke`.
