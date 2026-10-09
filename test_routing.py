"""Deterministic check of the folder routing — no Ollama, no model, no images.

    python test_routing.py

The chat lane used to send screenshots to bad_quality/, so the two lanes
produced different folder sets. This checks the decision table that fixes it,
plus the clef lane it has to stay consistent with. No framework, no fixtures.
"""
import importlib.util
import os
import tempfile
from pathlib import Path

SRC = Path(__file__).with_name("sort-images.py")


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

clef = load("clef")
assert clef.pick_folder({"screenshot"}) == clef.SCREENSHOT_FOLDER
assert clef.pick_folder({"photo_of_screen"}) == clef.SCREENSHOT_FOLDER
assert clef.pick_folder({"document"}) == clef.BAD_FOLDER
assert clef.pick_folder({"photo"}) == clef.OK_FOLDER

print("ok — 10 routing assertions passed")
