"""End-to-end check of sort_images() against a fake model — no Ollama, no images.

    python test_sort_run.py

The chat lane hands the *path* to the model and never opens the file, so empty
files plus a stub client exercise the whole loop: classify -> route -> move.
The run is repeated with WORKERS=4 and must give the identical result — a
worker pool that drops, duplicates or misroutes an image fails here.
"""
import importlib.util
import os
import tempfile
from pathlib import Path

SRC = Path(__file__).with_name("sort-images.py")

# Filename prefix -> what the fake model answers.
ANSWERS = {
    "screenshot": "screenshot",
    "blurry": "blurry, low quality",
    "sharp": "photo",
}
KINDS = 4


class FakeClient:
    """Stands in for ollama.Client: answers from the image path it is given."""

    def __init__(self):
        self.seen = []

    def chat(self, model, messages):
        image = messages[0]["images"][0]
        path = Path(getattr(image, "value", image))
        self.seen.append(path.name)
        for prefix, answer in ANSWERS.items():
            if path.name.startswith(prefix):
                return {"message": {"content": answer}}
        raise AssertionError(f"oväntad fil: {path.name}")


def build(workers: int):
    folder = Path(tempfile.mkdtemp(prefix="sorter-run-"))
    os.environ.update(
        CLASSIFIER="chat", INPUT_FOLDER=str(folder), WORKERS=str(workers),
        SECOND_OPINION="", MODEL_NAME="fake", OLLAMA_HOST="http://127.0.0.1:1",
    )
    spec = importlib.util.spec_from_file_location(f"sorter_w{workers}", SRC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for prefix in ANSWERS:
        for i in range(KINDS):
            (folder / f"{prefix}-{i}.jpg").write_bytes(b"")
    (folder / "notes.txt").write_text("inte en bild")

    client = FakeClient()
    setattr(module, "client", client)
    return module, folder, client


expected = {"screenshots": KINDS, "bad_quality": KINDS, "ok": KINDS}

for workers in (1, 4):
    module, folder, client = build(workers)
    module.sort_images()

    got = {name: len(list((folder / name).iterdir()))
           for name in expected}
    left = sorted(p.name for p in folder.iterdir() if p.is_file())

    assert got == expected, f"WORKERS={workers}: {got} != {expected}"
    assert left == ["notes.txt"], f"WORKERS={workers}: kvar i källmappen: {left}"
    assert sorted(client.seen) == sorted(
        f"{prefix}-{i}.jpg" for prefix in ANSWERS for i in range(KINDS)
    ), f"WORKERS={workers}: varje bild ska klassas exakt en gång"
    print(f"WORKERS={workers}: {got}")

print("ok — 1 and 4 workers give the same three folders, nothing lost")
