#!/usr/bin/env python3
"""clef_bench.py - mat Cloudflare Clef / Clef-flash mot ett facit.

Frågan scriptet svarar på: hittar en lokal vision-decision-modell suddiga
bilder och skärmdumpar bättre eller sämre än sort-images.py / fotorens
heuristik?

Clef är ingen chattmodell. Den tar en `state` + ett schema av typade frågor
och returnerar en sannolikhet per svarsalternativ i ett enda icke-
autoregressivt pass, via Ollama's /v1/systemone-endpoint (Ollama >= 0.35.1).

    pip install requests pillow python-dotenv
    ollama pull clef-flash          # 9B, ~11 GB
    ollama serve

    # facit från en fotorens-rapport (auto om _Rensat_rapport.json finns)
    python clef_bench.py "C:\\bilder" --limit 200 --resume

    # eller facit från mappnamn: screenshots/  blurry/  ok/
    python clef_bench.py "C:\\bilder\\urval" --facit-mode folders

Utdata:
    clef_bench_results.jsonl    en rad per bild (återupptas med --resume)
    clef_bench_summary.md       TP/FP/FN/TN, precision/recall/F1 mot facit
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import requests
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Saknar paket: {exc}. Kör: pip install requests pillow python-dotenv")

# Pillow >= 9.1 har Resampling-enum; äldre versioner har bara Image.LANCZOS.
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE = Image.LANCZOS

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv är frivilligt
    pass

DEFAULT_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("CLEF_MODEL_NAME", "clef-flash")

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
# Karantänmappen från fotorens hoppas över som standard (annars dubbelräknas
# redan flaggade bilder). Övriga mappar läses — annars fungerar inte
# --facit-mode folden, där hela poängen är att läsa screenshots/ blurry/ ok/.
SKIP_DIRS = {"_Rensat"}

REPORT_NAME = "_Rensat_rapport.json"

# Mappnamn -> facit, för --facit-mode folders
SCREENSHOT_HINTS = ("screenshot", "skarmdump", "skärmdump", "skarmbild")
BLURRY_HINTS = ("blurry", "oskarp", "oskärpa", "oskarpa", "bad_quality", "bad", "dalig", "dålig")

# Frågorna speglar sort-images.py's kategorier + en som heuristiken inte klarar.
QUESTIONS = {
    "image_type": {
        "type": "choice",
        "instructions": "Which category best describes this image?",
        "criteria": {
            "photo": "A photograph of a real scene, people or objects, taken with a camera.",
            "screenshot": "A screen capture of a phone or computer user interface.",
            "photo_of_screen": "A photograph taken with a camera pointing at a screen or monitor.",
            "document": "A scan or close-up of a document, receipt, form or printed text.",
            "other": "None of the above.",
        },
    },
    "blurry": {
        "type": "noul",
        "instructions": (
            "Is the main subject of this image out of focus, motion blurred, or "
            "otherwise too soft to be worth keeping?"
        ),
        "criteria": {
            "true": "The image is visibly unsharp, blurry or smeared.",
            "false": "The image is sharp enough to keep.",
        },
    },
    "keep": {
        "type": "noul",
        "instructions": (
            "Should this image be kept, rather than deleted or quarantined as "
            "a screenshot or a bad-quality photo?"
        ),
        "criteria": {
            "true": "The image is a real, sharp photograph worth keeping.",
            "false": "The image is a screenshot, a document, or too blurry to keep.",
        },
    },
}


def encode_image(path: Path, max_px: int) -> str:
    """Skala ned och JPEG-koda (Clef tar PNG/JPEG/WebP, max 4 MiB / 16 Mpx per bild)."""
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), RESAMPLE)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def classify(host: str, model: str, b64: str, timeout: int, keep_alive: str) -> dict:
    body = {
        "model": model,
        "state": "Classify this image.",
        "images": [b64],
        "questions": QUESTIONS,
        "keep_alive": keep_alive,
    }
    r = requests.post(host.rstrip("/") + "/v1/systemone", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------
# Facit
# --------------------------------------------------------------------------

def facit_from_report(root: Path, report: Path) -> dict[str, dict]:
    """Läs en fotorens-rapport: {'flaggade': [{'fil','status',...}]}."""
    data = json.loads(report.read_text(encoding="utf-8"))
    facit: dict[str, dict] = {}
    for item in data.get("flaggade", []):
        status = item.get("status", "")
        facit[item["fil"]] = {
            "screenshot": "screenshot" in status,
            "blurry": "oskarp" in status,
            "oskarpa_score": item.get("oskarpa_score"),
        }
    return facit


def facit_from_folders(root: Path, images: list[str]) -> dict[str, dict]:
    """Härled facit ur sökvägen: screenshots/ -> skärmdump, blurry/ -> oskarp."""
    facit: dict[str, dict] = {}
    for rel in images:
        parts = [p.lower() for p in Path(rel).parts[:-1]]
        screenshot = any(any(h in p for h in SCREENSHOT_HINTS) for p in parts)
        blurry = any(any(h in p for h in BLURRY_HINTS) for p in parts)
        facit[rel] = {"screenshot": screenshot, "blurry": blurry, "oskarpa_score": None}
    return facit


def load_facit(root: Path, images: list[str], mode: str, facit_path: Path | None) -> tuple[dict[str, dict], str]:
    """Returnerar (facit, etikett för vilken källa som användes)."""
    report = facit_path or (root / REPORT_NAME)
    use_report = mode == "report" or (mode == "auto" and report.exists())
    if use_report:
        if not report.exists():
            sys.exit(f"--facit-mode report men hittar inte {report}.")
        return facit_from_report(root, report), f"rapport ({report.name})"
    return facit_from_folders(root, images), "mappnamn (screenshots/blurry/ok)"


def collect_images(root: Path) -> list[str]:
    """Alla bilder under root, men inte inuti karantän-/mål-mappar."""
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        out.append(str(p.relative_to(root)))
    return sorted(out)


# --------------------------------------------------------------------------
# Analys
# --------------------------------------------------------------------------

def analyze_one(args, root: Path, rel: str, facit: dict[str, dict]) -> dict:
    default = {"screenshot": False, "blurry": False, "oskarpa_score": None}
    row: dict = {"fil": rel, "facit": facit.get(rel, default)}
    t0 = time.perf_counter()
    try:
        b64 = encode_image(root / rel, args.max_px)
        resp = classify(args.host, args.model, b64, args.timeout, args.keep_alive)
        answers = resp.get("answers", {})
        itype = answers.get("image_type", {})
        choice = itype.get("choice")
        blur = answers.get("blurry", {}).get("noul") or 0.0
        keep = answers.get("keep", {}).get("noul") or 0.0
        row["clef"] = {
            "image_type": choice,
            "image_type_prob": round(itype.get("probabilities", {}).get(choice, 0.0), 3),
            "screenshot": choice == "screenshot",
            "screenshot_or_screen_photo": choice in ("screenshot", "photo_of_screen"),
            "blurry": blur >= args.blur_threshold,
            "blurry_prob": round(blur, 3),
            "keep": keep >= 0.5,
            "keep_prob": round(keep, 3),
        }
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        if args.raw:
            row["raw"] = resp
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return row


def confusion(rows: list[dict], clef_key: str, facit_key: str) -> dict:
    tp = fp = fn = tn = 0
    fp_list, fn_list = [], []
    for r in rows:
        if "error" in r:
            continue
        c = bool(r["clef"][clef_key])
        f = bool(r["facit"][facit_key])
        if c and f:
            tp += 1
        elif c and not f:
            fp += 1
            fp_list.append(r["fil"])
        elif not c and f:
            fn += 1
            fn_list.append(r["fil"])
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    denom = prec + rec
    f1 = 2 * prec * rec / denom if denom == denom and denom > 0 else float("nan")
    agree = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 3), "recall": round(rec, 3),
        "f1": round(f1, 3), "agreement": round(agree, 3),
        "false_positives": fp_list[:50], "false_negatives": fn_list[:50],
    }


def summarize(rows: list[dict], model: str, facit_mode: str) -> str:
    done = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    lat = sorted(r["latency_ms"] for r in done)
    lines = [f"# clef_bench - {model}", "", f"Facit-källa: {facit_mode}", ""]
    lines.append(f"Bilder testade: {len(rows)}  (lyckade: {len(done)}, fel: {len(errs)})")
    if lat:
        lines.append(
            f"Latens per bild: median {statistics.median(lat):.0f} ms, "
            f"p95 {lat[min(len(lat) - 1, int(len(lat) * 0.95))]:.0f} ms"
        )
    lines.append("")

    for label, ck, fk in (
        ("Skärmdumpar (facit: ja = skärmdump)", "screenshot", "screenshot"),
        ("Skärmdumpar, bred (räknar även foto-av-skärm)", "screenshot_or_screen_photo", "screenshot"),
        ("Oskärpa (facit: ja = oskarp)", "blurry", "blurry"),
    ):
        c = confusion(done, ck, fk)
        lines.append(f"## {label}")
        lines.append(
            f"- TP {c['tp']} / FP {c['fp']} / FN {c['fn']} / TN {c['tn']}"
            f"  -> precision {c['precision']}, recall {c['recall']}, F1 {c['f1']},"
            f" samstämmighet {c['agreement']}"
        )
        if c["false_positives"]:
            lines.append(f"- Clef flaggade men facit sa OK ({len(c['false_positives'])} visade): "
                         + ", ".join(c["false_positives"][:10]))
        if c["false_negatives"]:
            lines.append(f"- facit flaggade men Clef sa OK ({len(c['false_negatives'])} visade): "
                         + ", ".join(c["false_negatives"][:10]))
        lines.append("")

    cats: dict[str, int] = {}
    for r in done:
        t = r["clef"]["image_type"] or "(tomt)"
        cats[t] = cats.get(t, 0) + 1
    lines.append("## Clef's egna kategorier")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")

    if errs:
        lines.append("")
        lines.append("## Fel")
        for r in errs[:10]:
            lines.append(f"- {r['fil']}: {r['error']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark Clef / Clef-flash mot ett facit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exempel:\n"
            "  python clef_bench.py Bilder --limit 200 --resume\n"
            "  python clef_bench.py Bilder/urval --facit-mode folders\n"
            "  python clef_bench.py Bilder --facit C:/annan/_Rensat_rapport.json\n"
        ),
    )
    ap.add_argument("folder", type=Path, help="Bildmapp att analysera")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama-URL (default: {DEFAULT_HOST})")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"clef-flash eller clef:27b (default: {DEFAULT_MODEL})")
    ap.add_argument("--facit-mode", choices=["auto", "report", "folders"], default="auto",
                    help="auto = rapport om den finns, annars mappnamn")
    ap.add_argument("--facit", type=Path, default=None,
                    help="Sökväg till _Rensat_rapport.json om den ligger utanför mappen")
    ap.add_argument("--limit", type=int, default=200, help="Max antal bilder (0 = alla)")
    ap.add_argument("--sample-ok", type=int, default=0,
                    help="Extra: slumpa N icke-flaggade bilder som motprov")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallella anrop (kräver OLLAMA_NUM_PARALLEL)")
    ap.add_argument("--blur-threshold", type=float, default=0.5,
                    help="Sannolikhet över vilken bilden räknas som oskarp (default 0.5)")
    ap.add_argument("--max-px", type=int, default=1280, help="Nedskalning, längsta sida")
    ap.add_argument("--timeout", type=int, default=180, help="Timeout per bild (s)")
    ap.add_argument("--keep-alive", default="10m", help="Hur länge modellen ligger kvar i VRAM")
    ap.add_argument("--out", type=Path, default=None, help="Utdata-prefix (default: cwd)")
    ap.add_argument("--resume", action="store_true", help="Hoppa över bilder som redan finns i JSONL")
    ap.add_argument("--raw", action="store_true", help="Spara hela svaret per bild i JSONL")
    ap.add_argument("--smoke", action="store_true", help="Kör en bild och skriv ut rå JSON")
    args = ap.parse_args()

    root = args.folder.resolve()
    if not root.is_dir():
        sys.exit(f"Inte en mapp: {root}")

    all_images = collect_images(root)
    if not all_images:
        sys.exit(f"Inga bilder hittades under {root}")

    if args.smoke:
        b64 = encode_image(root / all_images[0], args.max_px)
        print(json.dumps(classify(args.host, args.model, b64, args.timeout, args.keep_alive),
                         indent=2, ensure_ascii=False))
        return

    facit, mode_used = load_facit(root, all_images, args.facit_mode, args.facit)

    flagged = [f for f in all_images if facit.get(f, {}).get("screenshot") or facit.get(f, {}).get("blurry")]
    ok = [f for f in all_images if f not in flagged]

    selected = list(flagged)
    if args.limit and len(selected) > args.limit:
        selected = random.sample(selected, args.limit)
    n_ok = args.sample_ok or (max(0, args.limit - len(selected)) if args.limit else 0)
    pool_ok = list(ok)
    random.shuffle(pool_ok)
    selected += pool_ok[:n_ok]

    print(f"Facit: {len(flagged)} flaggade, {len(ok)} OK. Testar {len(selected)} bilder "
          f"({len(selected) - len([s for s in selected if s in flagged])} motprov från OK).")

    jsonl = Path(f"{args.out}_clef_bench_results.jsonl") if args.out else Path("clef_bench_results.jsonl")
    summary_path = jsonl.with_name(jsonl.name.replace("_results.jsonl", "_summary.md"))

    done_files: set[str] = set()
    rows: list[dict] = []
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done_files.add(r["fil"])
        print(f"Återupptar: {len(done_files)} bilder redan klara.")

    todo = [s for s in selected if s not in done_files]

    def work(rel: str) -> dict:
        return analyze_one(args, root, rel, facit)

    with jsonl.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(work, todo), 1):
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if "error" in row:
                tag = f"FEL {row['error'][:60]}"
            else:
                c = row["clef"]
                tag = f"{c['image_type']:<16} oskarp={c['blurry_prob']:<5} behåll={c['keep_prob']}"
            print(f"[{i}/{len(todo)}] {row['fil']}  {tag}  {row['latency_ms']} ms")

    summary = summarize(rows, args.model, mode_used)
    summary_path.write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nRapport: {summary_path}\nRådata:  {jsonl}")


if __name__ == "__main__":
    main()
