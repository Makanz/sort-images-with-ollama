"""Deterministic check of the folder routing — no Ollama, no model, no images.

    python tests/test_routing.py

The chat lane used to send screenshots to bad_quality/, so the two lanes
produced different folder sets. This checks the decision table that fixes it,
plus the clef lane it has to stay consistent with. No framework, no fixtures.

Since the chat prompt became a closed vocabulary it also checks the parser and
that the vocabulary can actually route somewhere: measured on 35 real images,
16 (46 %) fell through to ok/ because the model described the picture instead of
answering, and no label could catch it.
"""
import importlib.util
import os
import tempfile
from pathlib import Path

# tests/ ligger under repo-roten, där sort-images.py bor.
SRC = Path(__file__).resolve().parents[1] / "sort-images.py"
EXPLAIN = Path(__file__).resolve().parents[1] / "explain_sort.py"


def load(classifier: str):
    """Import the sorter with its own empty INPUT_FOLDER, one per lane."""
    os.environ["INPUT_FOLDER"] = tempfile.mkdtemp(prefix="sorter-test-")
    os.environ["CLASSIFIER"] = classifier
    os.environ["SECOND_OPINION"] = ""
    spec = importlib.util.spec_from_file_location(f"sorter_{classifier}", SRC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat = load("chat")
assert chat.pick_folder({"screenshot"}) == chat.SCREENSHOT_FOLDER
assert chat.pick_folder({"photo of screen"}) == chat.SCREENSHOT_FOLDER
assert chat.pick_folder({"blurry"}) == chat.BAD_FOLDER
assert chat.pick_folder({"low resolution", "blurry"}) == chat.BAD_FOLDER
assert chat.pick_folder({"photo"}) == chat.OK_FOLDER
assert chat.pick_folder(set()) == chat.OK_FOLDER

# Den slutna vokabulären: understreckade etiketter måste matcha samma listor som
# de gamla blankstegsstavningarna gjorde.
assert chat.pick_folder({"low_resolution"}) == chat.BAD_FOLDER
assert chat.pick_folder({"low_quality"}) == chat.BAD_FOLDER
assert chat.pick_folder({"photo_of_screen"}) == chat.SCREENSHOT_FOLDER
# Skärmdump vinner fortfarande över dålig kvalitet.
assert chat.pick_folder({"screenshot", "blurry"}) == chat.SCREENSHOT_FOLDER
assert chat.pick_folder({"ok"}) == chat.OK_FOLDER

# Parsern: bara ord ur vokabulären räknas, "ok" är en sentinel, resten ignoreras.
assert chat.parse_categories("screenshot, blurry") == {"screenshot", "blurry"}
assert chat.parse_categories("Blurry, Low Resolution") == {"blurry", "low_resolution"}
assert chat.parse_categories("photo of screen") == {"photo_of_screen"}
assert chat.parse_categories("blurred image, slightly blurry") == {"blurry"}
assert chat.parse_categories("ok, screenshot") == {"screenshot"}
assert chat.parse_categories("menu, restaurant, food and drink") == set()
assert chat.parse_categories("") == set()
assert chat.parse_categories("blurry, blurry") == {"blurry"}
assert chat.unknown_words("menu, blurry") == ["menu"]

# Vokabulärintegriteten — det här är buggklassen vi fixar. Faller om en etikett
# som prompten erbjuder inte kan routa någonstans.
assert set(chat.CHAT_LABELS) - {"ok"} <= chat.BAD_NORMALIZED | chat.SCREENSHOT_NORMALIZED
assert set(chat.CHAT_LABEL_HELP) >= (set(chat.CHAT_LABELS) - {"ok"})
for label in chat.CHAT_LABELS:
    assert label in chat.CHAT_PROMPT, f"{label} saknas i prompten"

clef = load("clef")
assert clef.pick_folder({"screenshot"}) == clef.SCREENSHOT_FOLDER
assert clef.pick_folder({"photo_of_screen"}) == clef.SCREENSHOT_FOLDER
assert clef.pick_folder({"document"}) == clef.BAD_FOLDER
assert clef.pick_folder({"photo"}) == clef.OK_FOLDER

# Rapportens prompt måste vara sorterarens, annars slutar explain_sort att mäta
# det den påstår att den mäter. Kopiorna kan inte delas via import — se kommentaren
# i explain_sort.py — så de låses mot varandra här.
espec = importlib.util.spec_from_file_location("explain_prompt_check", EXPLAIN)
assert espec and espec.loader
explain = importlib.util.module_from_spec(espec)
espec.loader.exec_module(explain)
assert chat.CHAT_PROMPT == explain.CHAT_PROMPT, "promptkopian i explain_sort har glidit isär"
assert chat.CHAT_ALIASES == explain.CHAT_ALIASES, "aliastabellen har glidit isär"
assert chat.CLEF_QUESTIONS == explain.CLEF_QUESTIONS, "CLEF_QUESTIONS-kopian har glidit isär"

print("ok — routing, parser, vokabulär och promptkopior kontrollerade")
