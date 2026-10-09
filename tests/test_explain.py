"""Kontroll av explain_sort.py — ingen Ollama, inga riktiga bilder.

    python tests/test_explain.py

Två lager. Först de rena härledningarna (regelval, hink, avvikelse, HTML) som
tar sina argument explicit och därför kan testas utan miljö och utan omladdning
— till skillnad från test_routing.py, som måste ladda om modulen per lane.

Sedan en hel körning av main() mot en stubbad modell, enligt samma mönster som
test_sort_run.py: modulens globala ask_image byts ut, så varken Ollama eller en
riktig klassificering behövs. Bilderna måste däremot vara giltiga, eftersom
tumnaglarna kodas om på riktigt.
"""
import importlib.util
import io
import json
import os
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "explain_sort.py"

TMP = Path(tempfile.mkdtemp(prefix="explain-test-"))
os.environ.update(
    INPUT_FOLDER=str(TMP),
    CLASSIFIER="clef",
    SECOND_OPINION="",
    BLUR_THRESHOLD="0.9",
    OLLAMA_HOST="http://127.0.0.1:1",
)
spec = importlib.util.spec_from_file_location("explain_sort_test", SRC)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def meta() -> dict:
    return {
        "classifier": "clef", "model": "clef-flash", "host": "http://x:11434",
        "generated": "2026-01-01 00:00", "blur_threshold": 0.9, "low_conf": 0.8,
        "buckets": module.folder_names(), "alias": module.bucket_alias(),
        "folders": ["screenshots", "bad_quality"],
    }


def clef_row(fil: str, mapp: str, **kwargs) -> dict:
    row = {"fil": fil, "mapp": mapp, "classifier": "clef", "model": "clef-flash"}
    row["clef"] = {
        "image_type": kwargs.get("image_type", "photo"),
        "confidence": kwargs.get("confidence", 0.95),
        "probabilities": kwargs.get("probabilities", {"photo": 0.95}),
        "blur_prob": kwargs.get("blur_prob", 0.02),
    }
    return row


# ── Kategorier och regelval ──────────────────────────────────────────────────
assert module.clef_categories({"image_type": {"choice": "photo"},
                               "blurry": {"noul": 0.02}}, 0.9) == set()
assert module.clef_categories({"image_type": {"choice": "photo"},
                               "blurry": {"noul": 0.95}}, 0.9) == {"blurry"}
# Tröskeln är >=, inte >.
assert module.clef_categories({"image_type": {"choice": "photo"},
                               "blurry": {"noul": 0.9}}, 0.9) == {"blurry"}
assert module.clef_categories({"image_type": {"choice": "screenshot"},
                               "blurry": {"noul": 0.0}}, 0.9) == {"screenshot"}

# Fast ordning: low_resolution slår blurry, precis som pick_folder itererar.
assert module.clef_winner({"blurry", "low_resolution"}) == "low_resolution"
assert module.clef_winner({"screenshot", "blurry"}) == "screenshot"
assert module.clef_winner({"photo"}) is None
# "other" läggs till som kategori men matchar ingen regel -> ok.
assert module.clef_winner({"other"}) is None

assert module.chat_categories("blurry, low quality") == ["blurry", "low_quality"]
assert module.chat_categories("") == []
assert module.chat_categories("  Screenshot ,") == ["screenshot"]
# Sluten vokabulär: en beskrivning ger ingen kategori, och "blurred" räknas som
# "blurry" — samma parser som sorteraren använder.
assert module.chat_categories("menu, restaurant, food and drink") == []
assert module.chat_categories("blurred image") == ["blurry"]
# Skärmdump vinner över dålig kvalitet när ordet står i båda listorna.
assert module.chat_winner(["screenshot"], {"screenshot"}, {"screenshot", "blurry"}) == "screenshot"


# ── Avvikelser ───────────────────────────────────────────────────────────────
assert module.compare("ok", "ok") == (True, None)
assert module.compare("bad_quality", None) == (True, None)   # okänd mapp: inget att rapportera
assert module.compare("bad_quality", "screenshots") == (False, "cross")
assert module.compare("screenshots", "bad_quality") == (False, "cross")
assert module.compare("ok", "bad_quality") == (False, "to_ok")
assert module.compare("ok", "screenshots") == (False, "to_ok")

alias = module.bucket_alias()
assert module.bucket_of_current("screenshots/a.jpg", alias) == "screenshots"
assert module.bucket_of_current("bad_quality/djupt/b.jpg", alias) == "bad_quality"
assert module.bucket_of_current("ok/c.jpg", alias) == "ok"
assert module.bucket_of_current("lös/bild.jpg", alias) is None


# ── Förklaringen ─────────────────────────────────────────────────────────────
# En bild i bad_quality/ som modellen nu kallar low_resolution: skälet ska namnge
# regeln, dess sannolikhet och tröskeln den jämfördes mot.
info = module.explain_row(
    clef_row("bad_quality/1.jpg", "bad_quality", image_type="low_resolution",
             confidence=0.62, probabilities={"photo": 0.2, "low_resolution": 0.71},
             blur_prob=0.31),
    blur_threshold=0.9, alias=alias, low_conf=0.8)
assert info["rule"] == "low_resolution", info["rule"]
assert info["implied"] == "bad_quality"
assert info["match"] and info["kind"] is None
assert "low_resolution" in info["reason"] and "0.71" in info["reason"]
assert "0.90" in info["reason"], info["reason"]
assert info["low_conf"] is True, "confidence 0.62 ligger under 0.8"

# Samma bild, men modellen säger nu foto -> den värdefulla avvikelsen.
info = module.explain_row(
    clef_row("bad_quality/2.jpg", "bad_quality", image_type="photo",
             probabilities={"photo": 0.9}),
    blur_threshold=0.9, alias=alias, low_conf=0.8)
assert info["implied"] == "ok" and info["kind"] == "to_ok", info

# En felrad ska inte krascha härledningen.
info = module.explain_row({"fil": "bad_quality/3.jpg", "mapp": "bad_quality",
                           "classifier": "clef", "error": "HTTPError: 500"},
                          blur_threshold=0.9, alias=alias, low_conf=0.8)
assert info["error"] and info["reason"] == ""


# ── HTML ──────────────────────────────────────────────────────────────────────
rows = [
    clef_row("screenshots/ren.png", "screenshots", image_type="screenshot",
             probabilities={"photo": 0.02, "screenshot": 0.93}),
    clef_row("screenshots/konstig.png", "screenshots", image_type="photo",
             confidence=0.33, probabilities={"photo": 0.61}, blur_prob=0.04),
    clef_row("bad_quality/suddig.jpg", "bad_quality", image_type="photo",
             probabilities={"photo": 0.8}, blur_prob=0.97),
    # Skräp i fel skräpmapp: skärmdump som hamnat i bad_quality -> "cross".
    clef_row("bad_quality/skarmdump.png", "bad_quality", image_type="screenshot",
             probabilities={"screenshot": 0.88}),
    {"fil": "screenshots/<script>alert(1)</script>.png", "mapp": "screenshots",
     "classifier": "clef", "error": "UnidentifiedImageError: trasig fil"},
]
page = module.render_html(rows, {"screenshots/ren.png": "data:image/jpeg;base64,AAAA"},
                          meta())
assert page.startswith("<!doctype html>"), page[:40]
assert 'id="screenshots"' in page and 'id="bad_quality"' in page
assert '"badge to_ok">TILL OK' in page, "foto i bad_quality ska flaggas till ok"
assert '"badge cross">AVVIKER' in page, "skärmdump i bad_quality ska flaggas som cross"
assert '"badge error">FEL' in page
assert "data:image/jpeg;base64,AAAA" in page
assert "&lt;script&gt;" in page and "<script>alert" not in page, "filnamn ska escapas"
# Sektionen för bad_quality ska bara innehålla sin egen bild.
assert page.count("<article") == 5, page.count("<article")
# Avvikelserna ska sorteras före de som matchar.
assert page.index("badge to_ok") < page.index("badge match"), "avvikelser först"

print("ok — härledningar, avvikelser och HTML")

# ── Hela körningen, med stubbad modell ────────────────────────────────────────
from PIL import Image  # noqa: E402 — modulen kräver Pillow ändå

buf = io.BytesIO()
Image.new("RGB", (8, 8), (120, 30, 30)).save(buf, "PNG")
png = buf.getvalue()

(TMP / "screenshots").mkdir(parents=True, exist_ok=True)
(TMP / "bad_quality").mkdir(parents=True, exist_ok=True)
for name in ("a.png", "b.png", "c.png"):
    (TMP / "screenshots" / name).write_bytes(png)
for name in ("x.jpg", "y.jpg"):
    (TMP / "bad_quality" / name).write_bytes(png)
(TMP / "screenshots" / "anteckning.txt").write_text("inte en bild")


def fake_ask_image(args, path: Path) -> dict:
    """screenshots/ kallas skärmdump (matchar), bad_quality/ kallas skarpt foto
    (avviker till ok)."""
    if path.parent.name == "screenshots":
        return {"answers": {"image_type": {"choice": "screenshot", "confidence": 0.9,
                                           "probabilities": {"screenshot": 0.9}},
                            "blurry": {"noul": 0.01}}}
    return {"answers": {"image_type": {"choice": "photo", "confidence": 0.4,
                                       "probabilities": {"photo": 0.7}},
                        "blurry": {"noul": 0.02}}}


module.ask_image = fake_ask_image
out = TMP / "out"
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys  # noqa: E402

sys.argv = ["explain_sort.py", str(TMP), "--classifier", "clef", "--out", str(out),
            "--blur-threshold", "0.9"]
module.main()

jsonl = Path(f"{out}_explain_results.jsonl")
html_path = Path(f"{out}_explain_report.html")
assert jsonl.exists() and html_path.exists(), "båda utfilerna ska skrivas"

rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 5, f"5 bilder, inte {len(rows)} (txt-filen ska hoppas över)"
assert all("clef" in r and "latency_ms" in r for r in rows)
assert all(r["classifier"] == "clef" for r in rows)

page = html_path.read_text(encoding="utf-8")
# Alla fem ligger i en skräpmapp men kallas ok/screenshot -> bad_quality-bilderna
# ska flaggas som avvikelser, screenshots-bilderna ska matcha.
assert page.count("<article") == 5
assert "TILL OK" in page, "bad_quality-bilderna ska flaggas som avvikelse till ok"
assert page.count("MATCHAR") >= 3

# --resume ska inte fråga om något, och rapporten ska vara oförändrad.
before = rows
sys.argv = ["explain_sort.py", str(TMP), "--classifier", "clef", "--out", str(out), "--resume"]
module.main()
after = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(after) == len(before), f"--resume ska inte lägga till rader: {len(after)} != {len(before)}"

print("ok — 5 bilder routade, avvikelserna flaggade, --resume frågar inte om")
