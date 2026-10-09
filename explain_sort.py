#!/usr/bin/env python3
"""explain_sort.py — varför hamnade bilderna i screenshots/ och bad_quality/?

sort-images.py skriver aldrig ned varför en bild sorterades ut; varje skäl var
en print() till stdout och finns inte kvar. Det här scriptet svarar på frågan i
efterhand: det ställer samma fråga till modellen igen om varje bild som redan
ligger i mål­mapparna och skriver en HTML-sida där tumnageln står bredvid domen,
sannolikheterna och confidence.

Det intressanta är AVVIKELSEN: en bild som ligger i bad_quality/ men som
modellen nu kallar helt ok är antingen felSORTERAD eller ett tecken på att
domen var tveksam från början. En avvikelse betyder "modellen tycker något
annat nu" — inte "filen är fel". Chat-lanen är inte deterministisk, och ett
modellbyte går inte att skilja från en verklig felsortering.

Scriptet är en LÄSARE. Det flyttar, byter namn på eller raderar aldrig något i
bildträdet. Det skriver bara två filer: en JSONL-cache och en HTML-rapport.

    pip install -r requirements.txt
    python explain_sort.py                      # använder INPUT_FOLDER ur .env
    python explain_sort.py --classifier clef --resume
    python explain_sort.py --smoke               # en bild, rå modellutdata

Till skillnad från sort-images.py har --workers inga atomaritetskrav här:
sorteraren måste flytta i huvudtråden eftersom get_unique_path() + shutil.move
är en icke-atomär check-then-act. Det här scriptet flyttar ingenting, så både
modellanropen och tumnaglarna får köras i en trådpool. Bara JSONL-skrivningen
hålls i huvudtråden.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# Windows-konsolen är ofta cp1252 och kan inte koda emojin i utskrifterna nedan.
# Utan detta kraschar körningen på en print så snart stdout är en pipe eller en
# omdirigering — vilket är precis vad man gör när man loggar en lång körning.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # saknas i äldre Python, eller stängd utgång
        pass

try:
    import requests
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Saknar paket: {exc}. Kör: pip install -r requirements.txt")

# Pillow >= 9.1 har Resampling-enum; äldre versioner har bara Image.LANCZOS.
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE = Image.LANCZOS

# Telefonbilder är ofta HEIC. Pillow klarar dem inte utan detta, och felet ser
# ut som en trasig fil snarare än en saknad beroende.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv är frivilligt
    pass


# ── Konfiguration ──────────────────────────────────────────────────────────
# Läsningen sker vid import, precis som i sort-images.py, så att konstanterna
# går att monkeypatcha i tester. Men till skillnad från sorteraren: ingen
# sys.exit, ingen Client() och inga makedirs här — annars går modulen inte att
# importera. Validering och mappkontroller ligger i main().
INPUT_FOLDER = os.getenv("INPUT_FOLDER", "images")
SCREENSHOT_FOLDER_NAME = os.getenv("SCREENSHOT_FOLDER_NAME", "screenshots")
BAD_QUALITY_FOLDER_NAME = os.getenv("BAD_QUALITY_FOLDER_NAME", "bad_quality")
OK_QUALITY_FOLDER_NAME = os.getenv("OK_QUALITY_FOLDER_NAME", "ok")
CLASSIFIER = os.getenv("CLASSIFIER", "chat").strip().lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
CHAT_MODEL = os.getenv("MODEL_NAME", "gemma3:4b")
CLEF_MODEL = os.getenv("CLEF_MODEL_NAME", "clef-flash")
MAX_PX = int(os.getenv("CLEF_MAX_PX", "1280"))
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "0.5"))

raw_extensions = os.getenv("SUPPORTED_EXTENSIONS",
                           ".jpg,.jpeg,.png,.bmp,.webp,.heic,.heif")
SUPPORTED_EXTENSIONS = set(
    ext.strip().lower() for ext in raw_extensions.split(",") if ext.strip()
)

raw_bad_categories = os.getenv(
    "BAD_CATEGORIES", "screenshot,blurry,low resolution,low quality")
BAD_CATEGORIES = set(
    cat.strip().lower() for cat in raw_bad_categories.split(",") if cat.strip()
)
raw_screenshot_categories = os.getenv(
    "SCREENSHOT_CATEGORIES", "screenshot,photo of screen")
SCREENSHOT_CATEGORIES = set(
    cat.strip().lower() for cat in raw_screenshot_categories.split(",")
    if cat.strip()
)


def _plural(n: int) -> str:
    """"1 bild" men "2 bilder"."""
    return "bild" if n == 1 else "bilder"


def _norm_category(text: str) -> str:
    """Normalisera ett kategorinamn: gemener, trimmat, blanksteg -> understreck."""
    return text.strip().lower().replace(" ", "_")


# Jämförelserna sker mot normaliserade former. Sorterarens chat-lane matchar
# sina råa ord mot BAD_CATEGORIES som stavar "low resolution" med blanksteg,
# medan clef-lanen skriver "low_resolution". Att normalisera båda sidor här är
# en avsiktlig avvikelse som bara påverkar just den stavningen.
BAD_NORMALIZED = {_norm_category(c) for c in BAD_CATEGORIES}
SCREENSHOT_NORMALIZED = {_norm_category(c) for c in SCREENSHOT_CATEGORIES}

# Frågeschemat är kopierat ordagrant från sort-images.py (CLEF_QUESTIONS), INKLUSIVE
# low_resolution. clef_bench.py's QUESTIONS saknar low_resolution och lägger till
# en keep-fråga; att använda den skulle göra "skulle hamna i" till en jämförelse
# som inte reproducerar vad sorteraren faktiskt gjorde.
CLEF_QUESTIONS = {
    "image_type": {
        "type": "choice",
        "instructions": "Which category best describes this image?",
        "criteria": {
            "photo": "A photograph of a real scene, people or objects, taken with a camera.",
            "screenshot": "A screen capture of a phone or computer user interface.",
            "photo_of_screen": "A photograph taken with a camera pointing at a screen or monitor.",
            "low_resolution": "A very small, pixelated or heavily compressed image.",
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
}

# Samma fasta ordning som pick_folder() itererar (sort-images.py:260). Att
# low_resolution står före blurry är inte godtyckligt: en bild som är både
# pixlig och oskarp styrs av low_resolution.
CLEF_PRECEDENCE = ("screenshot", "photo_of_screen",
                   "low_resolution", "document", "blurry")

# Kategori -> semantisk hink. Medvetet INTE mappnamn: CLEF_MOVE_MAP i
# sort-images.py hårdkodar "screenshots" i stället för SCREENSHOT_FOLDER_NAME,
# så clef-lanen ignorerar ett bytt mappnamn medan chat-lanen följer det. Att
# jämföra råa mappnamn skulle fylla sidan med spökavvikelser.
CLEF_BUCKET = {
    "screenshot": "screenshots",
    "photo_of_screen": "screenshots",
    "low_resolution": "bad_quality",
    "document": "bad_quality",
    "blurry": "bad_quality",
}

# Chat-lanens prompt. KOPIA av sort-images.py:s — de två filerna kan inte
# importera varandra (sort-images.py har bindestreck i filnamnet och kör
# sys.exit/Client()/makedirs vid import), så tests/test_routing.py jämför
# strängarna i stället. Glider de isär slutar rapporten att mäta sorteraren utan
# att säga till.
CHAT_LABELS = tuple(sorted(BAD_NORMALIZED | SCREENSHOT_NORMALIZED)) + ("ok",)

CHAT_LABEL_HELP = {
    "screenshot": "a screen capture of a phone or computer user interface",
    "photo_of_screen": "a photograph of a screen or monitor",
    "low_resolution": "tiny, pixelated or heavily compressed",
    "low_quality": "otherwise too poor to keep",
    "document": "a scan or close-up of a document, receipt or printed text",
    "blurry": "out of focus, motion blurred or smeared",
}

CHAT_PROMPT = (
    "You are an image quality classifier. Decide which of these labels apply.\n"
    "Labels (use ONLY these exact words): " + ", ".join(CHAT_LABELS) + ".\n"
    + "".join(f"- {label}: {CHAT_LABEL_HELP.get(label, 'applies to this image')}.\n"
              for label in CHAT_LABELS if label != "ok")
    + "- ok: nothing above applies. Never combine ok with another label.\n"
    "Reply with ONE line: the applicable labels separated by commas.\n"
    "Do not describe the image. Do not explain."
)

# Kanoniskt namn per stavning modellen faktiskt använder. KOPIA av sorterarens
# tabell — samma test jämför dem.
CHAT_ALIASES = {
    "screenshot": "screenshot",
    "screen shot": "screenshot",
    "screen capture": "screenshot",
    "photo of screen": "photo_of_screen",
    "photo_of_screen": "photo_of_screen",
    "screen photo": "photo_of_screen",
    "low resolution": "low_resolution",
    "low_resolution": "low_resolution",
    "low quality": "low_quality",
    "low_quality": "low_quality",
    "blurry": "blurry",
    "blurred": "blurry",
    "blurry image": "blurry",
    "blurred image": "blurry",
    "slightly blurry": "blurry",
    "out of focus": "blurry",
    "document": "document",
    "receipt": "document",
    "scan": "document",
    "ok": "ok",
    "okay": "ok",
}

# Hink -> de mappnamn som räknas som den hinken. Standardnamnen står med även
# när de bytts, eftersom clef-lanens hårdkodade "screenshots" kan ha skapat en
# sådan mapp ändå.
BUCKET_FOLDERS = {
    "screenshots": {"screenshots"},
    "bad_quality": {"bad_quality"},
    "ok": {"ok"},
}


def folder_names() -> dict[str, str]:
    """Hink -> aktuellt mappnamn, ur miljön."""
    return {
        "screenshots": SCREENSHOT_FOLDER_NAME,
        "bad_quality": BAD_QUALITY_FOLDER_NAME,
        "ok": OK_QUALITY_FOLDER_NAME,
    }


def bucket_alias() -> dict[str, set[str]]:
    """Hink -> alla gemena mappnamn som hör till hinken."""
    names = folder_names()
    alias = {}
    for bucket, defaults in BUCKET_FOLDERS.items():
        alias[bucket] = defaults | {names[bucket].lower()}
    return alias


# ── Rena härledningar ───────────────────────────────────────────────────────
# Alla tar sina argument explicit, så de går att testa utan miljö, utan
# omladdning och utan Ollama.

def clef_categories(answers: dict, blur_threshold: float) -> set[str]:
    """Reproducerar classify_image_clef: image_type != photo läggs till,
    blurry läggs till när blurry.noul når upp till tröskeln."""
    categories: set[str] = set()
    itype = (answers.get("image_type") or {}).get("choice")
    if itype and itype != "photo":
        categories.add(itype)
    blur = (answers.get("blurry") or {}).get("noul") or 0.0
    if blur >= blur_threshold:
        categories.add("blurry")
    return categories


def clef_winner(categories: set[str]) -> str | None:
    """Regeln som avgjorde — första träffen i samma fasta ordning som pick_folder."""
    for cat in CLEF_PRECEDENCE:
        if cat in categories:
            return cat
    return None


def _answer_chunks(text: str) -> list[str]:
    """Samma uppdelning som sorterarens _answer_chunks."""
    return [" ".join(chunk.lower().strip().strip(".\"'()[]*_").split())
            for chunk in re.split(r"[,;\n]", str(text))]


def chat_categories(raw_text: str) -> list[str]:
    """Dela modellens svar i kanoniska kategorier, ordningen bevarad.

    Sluten vokabulär, precis som sorterarens parse_categories: bara ord ur
    CHAT_ALIASES räknas. Ett beskrivande svar ("wildlife trap, outdoor, rodent")
    ger en tom lista — det är själva fyndet, inte ett fel.
    """
    found: list[str] = []
    for chunk in _answer_chunks(raw_text):
        label = CHAT_ALIASES.get(chunk)
        if label and label not in found:
            found.append(label)
    return found


def chat_winner(categories, screenshot_categories, bad_categories) -> str | None:
    """Första kategori som matchar. Skärmdump vinner över dålig kvalitet, precis
    som pick_folder:s chat-gren (där kontrolleras SCREENSHOT_CATEGORIES först)."""
    for cat in categories:
        if cat in screenshot_categories:
            return cat
    for cat in categories:
        if cat in bad_categories:
            return cat
    return None


def bucket_of_current(rel: str, alias: dict[str, set[str]]) -> str | None:
    """Vilken hink bilden ligger i, utifrån sin sökväg relativt INPUT_FOLDER."""
    for part in Path(rel).parts[:-1]:
        for bucket, names in alias.items():
            if part.lower() in names:
                return bucket
    return None


def compare(implied: str, current: str | None) -> tuple[bool, str | None]:
    """(matchar, slag av avvikelse). Slaget är 'to_ok', 'cross' eller None."""
    if current is None or implied == current:
        # Okänd mapp: inget att rapportera som avvikelse.
        return True, None
    if implied == "ok":
        # Modellen säger "behåll" men bilden ligger i en skräpmapp. Det är den
        # värdefulla signalen: antingen verkligt felsorterad, eller tveksam dom.
        return False, "to_ok"
    # Skräp i fel skräpmapp — båda är ute, bara hinken är fel.
    return False, "cross"


def explain_row(row: dict, blur_threshold: float, alias: dict[str, set[str]],
                low_conf: float) -> dict:
    """Härled regel, målhink, orsak och avvikelse ur en sparad rad.

    Allt räknas om här i stället för när bilden frågades: då kan tröskeln ändras
    utan att modellen behöver frågas igen.
    """
    info = {
        "fil": row.get("fil", ""),
        "mapp": row.get("mapp") or bucket_of_current(row.get("fil", ""), alias),
        "error": row.get("error"),
        "rule": None,
        "categories": [],
        "implied": "ok",
        "detail": {},
        "reason": "",
        "low_conf": False,
        "match": True,
        "kind": None,
    }

    if row.get("error"):
        info["current"] = info["mapp"]
        return info

    if row.get("classifier") == "clef" or "clef" in row:
        c = row.get("clef") or {}
        itype = c.get("image_type")
        conf = c.get("confidence")
        blur = c.get("blur_prob")
        probs = c.get("probabilities") or {}
        answers = {
            "image_type": {"choice": itype, "confidence": conf, "probabilities": probs},
            "blurry": {"noul": blur},
        }
        categories = clef_categories(answers, blur_threshold)
        rule = clef_winner(categories)
        implied = "ok" if rule is None else CLEF_BUCKET[rule]

        p_win = probs.get(itype)
        detail = {
            "image_type": itype,
            "p_win": p_win,
            "confidence": conf,
            "blur_prob": blur,
            "probabilities": probs,
        }
        parts = [f"image_type={itype or '(tomt)'}"]
        if p_win is not None:
            parts.append(f"p={p_win:.2f}")
        if conf is not None:
            parts.append(f"confidence={conf:.2f}")
        head = ", ".join(parts)
        if blur is None:
            tail = "oskärpa okänd"
        else:
            rel = "≥" if blur >= blur_threshold else "<"
            tail = f"oskärpa {blur:.2f} {rel} tröskeln {blur_threshold:.2f}"
        reason = (f"{head}; {tail} → {implied}/ "
                  + (f"via regeln {rule}" if rule else "(ingen regel slog till)"))
        info["low_conf"] = conf is not None and conf < low_conf
    else:
        ch = row.get("chat") or {}
        answer = ch.get("answer", "")
        categories = chat_categories(answer)
        rule = chat_winner(categories, SCREENSHOT_NORMALIZED, BAD_NORMALIZED)
        if rule is None:
            implied = "ok"
        elif rule in SCREENSHOT_NORMALIZED:
            implied = "screenshots"
        else:
            implied = "bad_quality"
        detail = {"answer": answer, "categories": categories}
        reason = (f"modellen svarade ”{answer}” → {implied}/ "
                  + (f"via nyckelordet {rule}" if rule else "inget nyckelord matchade"))

    info.update(
        rule=rule,
        categories=sorted(categories),
        implied=implied,
        detail=detail,
        reason=reason,
        current=info["mapp"],
    )
    info["match"], info["kind"] = compare(implied, info["mapp"])
    return info


# ── Modellanrop ─────────────────────────────────────────────────────────────

def encode_image(path: Path, max_px: int, quality: int = 85) -> str:
    """Skala ned och JPEG-koda (Clef tar PNG/JPEG/WebP, max 4 MiB / 16 Mpx per bild)."""
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), RESAMPLE)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def ask_clef(args, path: Path) -> dict:
    """Typat schema via /v1/systemone — samma kropp som sort-images.py skickar."""
    body = {
        "model": args.model,
        "state": "Classify this image.",
        "images": [encode_image(path, args.max_px)],
        "questions": CLEF_QUESTIONS,
        "keep_alive": args.keep_alive,
    }
    r = requests.post(args.host.rstrip("/") + "/v1/systemone",
                      json=body, timeout=args.timeout)
    r.raise_for_status()
    return r.json()


def ask_chat(args, path: Path) -> dict:
    """Chat-lanen: samma prompt ordagrant som sort-images.py ställer."""
    body = {
        "model": args.model,
        "stream": False,
        "messages": [{"role": "user", "content": CHAT_PROMPT,
                      "images": [encode_image(path, args.max_px)]}],
    }
    r = requests.post(args.host.rstrip("/") + "/api/chat", json=body, timeout=args.timeout)
    r.raise_for_status()
    return r.json()


def ask_image(args, path: Path) -> dict:
    """Ställ samma fråga som sorteraren gjorde, i den valda lanen."""
    return ask_clef(args, path) if args.classifier == "clef" else ask_chat(args, path)


def analyze_one(args, root: Path, rel: str, bucket: str) -> dict:
    """Klassificera en bild och bygg en rad. Fel fångas per bild — en trasig
    HEIC eller en 500 från servern får inte fälla hela körningen."""
    row = {"fil": rel, "mapp": bucket, "classifier": args.classifier, "model": args.model}
    t0 = time.perf_counter()
    try:
        raw = ask_image(args, root / rel)
        if args.classifier == "clef":
            answers = raw.get("answers") or {}
            itype = answers.get("image_type") or {}
            row["clef"] = {
                "image_type": itype.get("choice"),
                "confidence": itype.get("confidence"),
                "probabilities": itype.get("probabilities") or {},
                "blur_prob": (answers.get("blurry") or {}).get("noul"),
            }
        else:
            text = str(((raw.get("message") or {}).get("content")) or "").strip().lower()
            row["chat"] = {"answer": text, "categories": chat_categories(text)}
        if args.raw:
            row["raw"] = raw
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return row


def build_thumbnails(rels, root: Path, thumb_px: int, workers: int) -> dict:
    """Tumnaglar som data-URI:er. Webbläsare kan inte visa HEIC, så de måste
    kodas om — samma väg som modellens nyttolast, bara mindre."""
    def one(rel: str):
        try:
            return rel, "data:image/jpeg;base64," + encode_image(
                root / rel, thumb_px, quality=78)
        except Exception:  # noqa: BLE001 — en bild utan tumnagel får en platshållare
            return rel, None

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(one, rels))
    return dict(one(rel) for rel in rels)


# ── HTML ────────────────────────────────────────────────────────────────────

def _bar_rows(probabilities: dict) -> list[tuple[str, float]]:
    """Alternativen i schemats ordning, plus eventuella extra nycklar."""
    keys = list(CLEF_QUESTIONS["image_type"]["criteria"])
    for key in probabilities:
        if key not in keys:
            keys.append(key)
    return [(key, probabilities.get(key, 0.0)) for key in keys]


def _bucket_title(meta: dict, bucket: str) -> str:
    return meta["buckets"].get(bucket, bucket)


def _card(info: dict, thumb: str | None, meta: dict) -> str:
    fil = html.escape(info["fil"])
    name = html.escape(Path(info["fil"]).name)

    if info["error"]:
        state, badge = "error", "FEL"
    elif info["kind"] == "to_ok":
        state, badge = "to_ok", "TILL OK"
    elif info["kind"] == "cross":
        state, badge = "cross", "AVVIKER"
    elif info["low_conf"]:
        state, badge = "lowconf", "LÅG"
    else:
        state, badge = "match", "MATCHAR"

    if thumb:
        img = (f'<img class="thumb" loading="lazy" alt="{name}" '
               f'src="{thumb}">')
    else:
        img = f'<div class="thumb placeholder">{name}</div>'

    if info["error"]:
        body = (f'<p class="verdict"><b>Kunde inte klassificeras</b></p>'
                f'<p class="reason">{html.escape(str(info["error"]))}</p>')
    else:
        cur = info["current"] or "?"
        arrow = (f'skulle hamna i <b>{html.escape(_bucket_title(meta, info["implied"]))}/</b>'
                 f' · ligger i <b>{html.escape(_bucket_title(meta, cur))}/</b>')
        bars = ""
        detail = info["detail"]
        if "probabilities" in detail:
            rows = []
            for label, value in _bar_rows(detail["probabilities"] or {}):
                win = " win" if label == detail.get("image_type") else ""
                pct = max(0.0, min(1.0, float(value or 0.0))) * 100
                rows.append(
                    f'<div class="prow{win}"><span class="plabel">'
                    f'{html.escape(label)}</span>'
                    f'<span class="bar"><i style="width:{pct:.1f}%"></i></span>'
                    f'<span class="pval">{float(value or 0.0):.2f}</span></div>')
            bars = f'<div class="probs">{"".join(rows)}</div>'
        body = (f'<p class="verdict">{arrow}</p>{bars}'
                f'<p class="reason">{html.escape(info["reason"])}</p>')

    return (f'<article class="card {state}">'
            f'<div class="shot">{img}</div>'
            f'<div class="meta"><h3 class="file" title="{fil}">{name}'
            f'<span class="badge {state}">{badge}</span></h3>{body}</div>'
            f'</article>')


def render_html(rows: list[dict], thumbs: dict, meta: dict) -> str:
    """Bygg hela sidan. Ren funktion: ingen I/O, inga modellanrop."""
    infos = [explain_row(r, meta["blur_threshold"], meta["alias"], meta["low_conf"])
             for r in rows]

    def order(info: dict):
        if info["error"]:
            rank = 3
        elif info["kind"] == "to_ok":
            rank = 0
        elif info["kind"] == "cross":
            rank = 1
        elif info["low_conf"]:
            rank = 2
        else:
            rank = 4
        return (rank, info["fil"])

    avviker = [i for i in infos if not i["error"] and i["kind"]]
    to_ok = [i for i in infos if i["kind"] == "to_ok"]
    lag = [i for i in infos if i["low_conf"]]
    fel = [i for i in infos if i["error"]]

    counts = (f'<ul class="counts">'
              f'<li class="match"><b>{len(infos) - len(avviker) - len(fel)}</b> matchar</li>'
              f'<li class="{"to_ok" if to_ok else "cross"}"><b>{len(avviker)}</b> avviker'
              + (f' (varav <b>{len(to_ok)}</b> till ok)' if to_ok else '') + '</li>'
              f'<li class="lowconf"><b>{len(lag)}</b> låg confidence</li>'
              f'<li class="error"><b>{len(fel)}</b> fel</li>'
              f'</ul>')

    sections = []
    for bucket in meta["folders"]:
        group = sorted([i for i in infos if i["mapp"] == bucket], key=order)
        if not group:
            continue
        n_av = len([i for i in group if not i["error"] and i["kind"]])
        cards = "".join(_card(i, thumbs.get(i["fil"]), meta) for i in group)
        sections.append(
            f'<section class="folder" id="{html.escape(bucket)}">'
            f'<h2>{html.escape(_bucket_title(meta, bucket))}/ '
            f'<span>{len(group)} {_plural(len(group))} · {n_av} avviker</span></h2>'
            f'<div class="cards">{cards}</div></section>')

    scanned = ", ".join(f"{_bucket_title(meta, b)}/" for b in meta["folders"])
    return f"""<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Varför sorterades bilderna ut?</title>
<style>
:root {{
  --bg:#f5f6f8; --fg:#15171c; --muted:#5b6472; --card:#fff; --line:#e2e5ea;
  --accent:#2563eb; --match:#1a7f47; --to_ok:#c0392b; --cross:#b5730a;
  --lowconf:#8a6d00; --error:#6b3fa0;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg:#131519; --fg:#e7e9ee; --muted:#98a1af; --card:#1b1e24; --line:#2a2f37;
    --accent:#6f9bff; --match:#4cc27e; --to_ok:#ff7b6b; --cross:#e0a63c;
    --lowconf:#d9bd4a; --error:#b393ff;
  }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2rem 1rem; background:var(--bg); color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
main {{ max-width:1180px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .4rem; }}
h2 {{ font-size:1.1rem; margin:2.2rem 0 .8rem; display:flex; align-items:baseline; gap:.6rem; }}
h2 span {{ font-size:.8rem; font-weight:400; color:var(--muted); }}
.meta {{ color:var(--muted); font-size:.82rem; margin:0 0 1rem; }}
.caveat {{ color:var(--muted); font-size:.82rem; border-left:3px solid var(--line);
  padding:.5rem .8rem; margin:1rem 0 0; }}
.counts {{ list-style:none; display:flex; flex-wrap:wrap; gap:.5rem; padding:0; margin:1rem 0 0; }}
.counts li {{ background:var(--card); border:1px solid var(--line); border-radius:999px;
  padding:.3rem .8rem; font-size:.82rem; color:var(--muted); }}
.counts li b {{ color:var(--fg); }}
.counts li.to_ok {{ border-color:var(--to_ok); color:var(--to_ok); }}
.counts li.to_ok b {{ color:var(--to_ok); }}
.counts li.lowconf {{ border-color:var(--lowconf); color:var(--lowconf); }}
.counts li.error {{ border-color:var(--error); color:var(--error); }}
.cards {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); }}
.card {{ display:flex; gap:.9rem; background:var(--card); border:1px solid var(--line);
  border-left:5px solid var(--line); border-radius:10px; padding:.8rem; align-items:flex-start; }}
.card.match {{ border-left-color:var(--match); }}
.card.to_ok {{ border-left-color:var(--to_ok); }}
.card.cross {{ border-left-color:var(--cross); }}
.card.lowconf {{ border-left-color:var(--lowconf); }}
.card.error {{ border-left-color:var(--error); }}
.shot {{ flex:0 0 116px; }}
.thumb {{ width:116px; height:116px; object-fit:cover; border-radius:8px;
  background:var(--line); display:block; }}
.thumb.placeholder {{ display:flex; align-items:center; justify-content:center;
  font-size:.62rem; color:var(--muted); text-align:center; padding:.3rem; word-break:break-all; }}
.meta {{ min-width:0; flex:1; }}
.file {{ font-size:.85rem; margin:0 0 .35rem; display:flex; align-items:center;
  gap:.45rem; flex-wrap:wrap; }}
.file {{ overflow-wrap:anywhere; }}
.badge {{ font-size:.62rem; letter-spacing:.04em; padding:.12rem .45rem; border-radius:999px;
  border:1px solid currentColor; white-space:nowrap; }}
.badge.match {{ color:var(--match); }}
.badge.to_ok {{ color:var(--to_ok); }}
.badge.cross {{ color:var(--cross); }}
.badge.lowconf {{ color:var(--lowconf); }}
.badge.error {{ color:var(--error); }}
.verdict {{ font-size:.8rem; margin:.1rem 0 .5rem; color:var(--muted); }}
.verdict b {{ color:var(--fg); }}
.probs {{ display:grid; gap:.15rem; margin:0 0 .5rem; }}
.prow {{ display:grid; grid-template-columns:7.5rem 1fr 2.2rem; align-items:center; gap:.45rem;
  font-size:.7rem; color:var(--muted); }}
.prow.win {{ color:var(--fg); }}
.bar {{ background:var(--line); border-radius:4px; height:.5rem; overflow:hidden; }}
.bar i {{ display:block; height:100%; background:var(--muted); }}
.prow.win .bar i {{ background:var(--accent); }}
.pval {{ text-align:right; font-variant-numeric:tabular-nums; }}
.reason {{ font-size:.72rem; color:var(--muted); margin:0; overflow-wrap:anywhere; }}
</style>
</head>
<body><main>
<h1>Varför hamnade bilderna i {html.escape(scanned)}?</h1>
<p class="meta">Lane <b>{html.escape(meta['classifier'])}</b> ·
modell <b>{html.escape(meta['model'])}</b> · {html.escape(meta['host'])} ·
oskärpa-tröskel {meta['blur_threshold']:.2f} · {len(infos)} {_plural(len(infos))} ·
{html.escape(meta['generated'])}</p>
{counts}
<p class="caveat">En avvikelse betyder att modellen svarar något annat nu än när bilden
sorterades — inte att filen är fel. Chat-lanen är inte deterministisk, och ett modellbyte
går inte att skilja från en verklig felsortering. Läs FP/FN-listan, inte bara siffrorna.</p>
{''.join(sections)}
</main></body></html>
"""


# ── Insamling och CLI ────────────────────────────────────────────────────────

def collect_images(folders: list[tuple[str, Path]], root: Path) -> list[tuple[str, str]]:
    """(rel, hink) för varje bild i mapparna, sorterade på sökväg."""
    out: list[tuple[str, str]] = []
    for bucket, folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                out.append((path.relative_to(root).as_posix(), bucket))
    return out


def load_existing(jsonl: Path) -> dict[str, dict]:
    """Läs JSONL:en, sista raden per fil vinner (en omfrågad bild ersätter en felrad)."""
    rows: dict[str, dict] = {}
    if not jsonl.exists():
        return rows
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row.get("fil", "")] = row
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Förklara varför bilder hamnade i screenshots/ och bad_quality/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exempel:\n"
            "  python explain_sort.py --resume\n"
            "  python explain_sort.py --classifier clef --include-ok --out rapport\n"
            "  python explain_sort.py --smoke\n"
        ),
    )
    ap.add_argument("folder", nargs="?", default=INPUT_FOLDER, type=Path,
                    help=f"Mapp att läsa (default: INPUT_FOLDER = {INPUT_FOLDER})")
    ap.add_argument("--classifier", choices=["chat", "clef"], default=CLASSIFIER,
                    help=f"Lanen som sorterade (default: {CLASSIFIER})")
    ap.add_argument("--folders", default=f"{SCREENSHOT_FOLDER_NAME},{BAD_QUALITY_FOLDER_NAME}",
                    help="Vilka undermappar som ska läsas (komma-separerat)")
    ap.add_argument("--include-ok", action="store_true",
                    help=f"Läs även {OK_QUALITY_FOLDER_NAME}/ som kontrollgrupp")
    ap.add_argument("--host", default=OLLAMA_HOST, help=f"Ollama-URL (default: {OLLAMA_HOST})")
    ap.add_argument("--model", default=None,
                    help=f"Modell (default: {CLEF_MODEL} för clef, {CHAT_MODEL} för chat)")
    ap.add_argument("--limit", type=int, default=0, help="Max antal bilder (0 = alla)")
    ap.add_argument("--workers", type=int, default=1, help="Parallella anrop (säkert här — inget flyttas)")
    ap.add_argument("--timeout", type=int, default=180, help="Timeout per bild (s)")
    ap.add_argument("--keep-alive", default="10m", help="Hur länge modellen ligger kvar i VRAM")
    ap.add_argument("--max-px", type=int, default=MAX_PX, help="Nedskalning till modellen, längsta sida")
    ap.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD,
                    help="Sannolikhet över vilken bilden räknas som oskarp")
    ap.add_argument("--low-conf", type=float, default=0.8,
                    help="Confidence under vilken kortet flaggas som lågt (default: 0.8)")
    ap.add_argument("--thumb-px", type=int, default=320, help="Tumnaglarnas längsta sida")
    ap.add_argument("--no-thumbs", action="store_true",
                    help="Textsida utan tumnaglar (liten sida även vid tusentals bilder)")
    ap.add_argument("--out", type=Path, default=None, help="Utdata-prefix (default: cwd)")
    ap.add_argument("--resume", action="store_true", help="Hoppa över bilder som redan är klara")
    ap.add_argument("--raw", action="store_true", help="Spara hela modellsvaret per bild")
    ap.add_argument("--smoke", action="store_true", help="Kör en bild och skriv ut rå JSON")
    return ap


def resolve_folders(args) -> tuple[list[tuple[str, Path]], list[str]]:
    """(mappar att läsa, varningar). Okända namn i --folders hoppas över."""
    alias = bucket_alias()
    wanted = [w.strip().lower() for w in args.folders.split(",") if w.strip()]
    if args.include_ok:
        wanted.append(OK_QUALITY_FOLDER_NAME.lower())

    folders, warnings = [], []
    for bucket, names in alias.items():
        if not any(w in names for w in wanted):
            continue
        folder = args.folder / folder_names()[bucket]
        if folder.is_dir():
            folders.append((bucket, folder))
        else:
            warnings.append(f"{folder} finns inte")
    return folders, warnings


def main() -> None:
    args = build_parser().parse_args()
    if args.model is None:
        args.model = CLEF_MODEL if args.classifier == "clef" else CHAT_MODEL
    if args.workers < 1:
        args.workers = 1

    root = Path(args.folder).resolve()
    if not root.is_dir():
        sys.exit(f"Inte en mapp: {root}\n"
                 f"Sätt INPUT_FOLDER i .env eller ange mappen som argument.")

    folders, warnings = resolve_folders(args)
    for warning in warnings:
        print(f"⚠️  {warning}")
    if not folders:
        sys.exit(f"Hittade inga av mapparna under {root}.\n"
                 f"Förväntade: {SCREENSHOT_FOLDER_NAME}/, {BAD_QUALITY_FOLDER_NAME}/ "
                 f"(och {OK_QUALITY_FOLDER_NAME}/ med --include-ok).")

    images = collect_images(folders, root)
    if not images:
        sys.exit(f"Inga bilder att förklara under {root}.")

    print(f"🔎 Förklarar {len(images)} {_plural(len(images))} "
          f"(lane: {args.classifier}, modell: {args.model})")

    if args.smoke:
        rel, _bucket = images[0]
        print(f"🔥 Smoke: {rel}")
        raw = ask_image(args, root / rel)
        print(json.dumps(raw, indent=2, ensure_ascii=False))
        return

    jsonl = Path(f"{args.out}_explain_results.jsonl") if args.out else Path("explain_results.jsonl")
    html_path = jsonl.with_name(jsonl.name.replace("_results.jsonl", "_report.html"))

    existing = load_existing(jsonl)
    todo = []
    for rel, bucket in images:
        cached = existing.get(rel)
        # En cachad rad återanvänds bara om den kom från samma lane och modell
        # och inte är en felrad — annars frågas bilden om.
        fresh = (cached and "error" not in cached
                 and cached.get("classifier") == args.classifier
                 and cached.get("model") == args.model)
        if not (args.resume and fresh):
            todo.append((rel, bucket))
    if args.resume and len(images) - len(todo):
        print(f"♻️  Återupptar: {len(images) - len(todo)} bilder redan klara.")

    if args.limit:
        todo = todo[:args.limit]
    buckets = {bucket for _rel, bucket in images}

    rows: dict[str, dict] = {}
    if todo:
        def work(item):
            return analyze_one(args, root, item[0], item[1])

        with jsonl.open("a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(work, todo), 1):
                rows[row["fil"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if "error" in row:
                    tag = f"FEL {row['error'][:60]}"
                else:
                    info = explain_row(row, args.blur_threshold, bucket_alias(), args.low_conf)
                    flag = {"to_ok": "AVVIKER (till ok)", "cross": "AVVIKER"}.get(info["kind"], "ok")
                    tag = f"{info['implied']:<12} {flag}"
                print(f"[{i}/{len(todo)}] {row['fil']}  {tag}  {row['latency_ms']} ms")

    # Rapporten byggs alltid av hela radmängden: cachade rader plus nya.
    final = []
    for rel, _bucket in images:
        row = rows.get(rel) or existing.get(rel)
        if row is not None:
            final.append(row)

    thumbs: dict = {}
    if not args.no_thumbs:
        thumbs = build_thumbnails([r["fil"] for r in final], root, args.thumb_px, args.workers)

    meta = {
        "classifier": args.classifier,
        "model": args.model,
        "host": args.host,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "blur_threshold": args.blur_threshold,
        "low_conf": args.low_conf,
        "buckets": folder_names(),
        "alias": bucket_alias(),
        "folders": [b for b, _f in folders] if folders else sorted(buckets),
    }
    html_path.write_text(render_html(final, thumbs, meta), encoding="utf-8")

    infos = [explain_row(r, args.blur_threshold, bucket_alias(), args.low_conf) for r in final]
    avviker = [i for i in infos if not i["error"] and i["kind"]]
    to_ok = [i for i in infos if i["kind"] == "to_ok"]
    fel = [i for i in infos if i["error"]]
    print(f"\n✅ {len(infos)} {_plural(len(infos))}: "
          f"{len(infos) - len(avviker) - len(fel)} matchar, "
          f"{len(avviker)} avviker (varav {len(to_ok)} till ok), {len(fel)} fel")
    print(f"\nRapport: {html_path}\nRådata:  {jsonl}")


if __name__ == "__main__":
    main()
