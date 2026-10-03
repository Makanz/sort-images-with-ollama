"""Sort images with a local Ollama model.

Two lanes, chosen with CLASSIFIER:

  chat   (default)  an ordinary chat model replies with a comma separated list
                    of categories. Simple, works with any model.
  clef              a decision model answers a typed schema in one pass via
                    Ollama's /v1/systemone endpoint (Ollama >= 0.35.1).
                    Returns calibrated probabilities instead of free text.

    pip install -r requirements.txt
    cp .env.example .env
    python sort-images.py
"""
import base64
import io
import os
import shutil
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from ollama import Client
from ollama._types import Image

try:
    from PIL import Image as PILImage
    try:
        RESAMPLE = PILImage.Resampling.LANCZOS
    except AttributeError:  # Pillow < 9.1
        RESAMPLE = PILImage.LANCZOS
    # Phone photos are often HEIC. Pillow can't open them without this.
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
except ImportError:
    PILImage = None

load_dotenv()

# 🧠 Konfiguration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://host.docker.internal:11434')
MODEL = os.getenv('MODEL_NAME', 'gemma3:4b')
CLEF_MODEL = os.getenv('CLEF_MODEL_NAME', 'clef-flash')
CLASSIFIER = os.getenv('CLASSIFIER', 'chat').strip().lower()
INPUT_FOLDER = os.getenv('INPUT_FOLDER', 'images')
BAD_QUALITY_FOLDER_NAME = os.getenv('BAD_QUALITY_FOLDER_NAME', 'bad_quality')
OK_QUALITY_FOLDER_NAME = os.getenv('OK_QUALITY_FOLDER_NAME', 'ok')
MAX_PX = int(os.getenv('CLEF_MAX_PX', '1280'))

BAD_FOLDER = os.path.join(INPUT_FOLDER, BAD_QUALITY_FOLDER_NAME)
OK_FOLDER = os.path.join(INPUT_FOLDER, OK_QUALITY_FOLDER_NAME)

# 📂 Stöd för bildformat
raw_extensions = os.getenv("SUPPORTED_EXTENSIONS",
                           ".jpg,.jpeg,.png,.bmp,.webp,.heic,.heif")
SUPPORTED_EXTENSIONS = set(
    ext.strip().lower() for ext in raw_extensions.split(",") if ext.strip()
)

# ❌ Kategorier för dålig bildkvalitet
raw_bad_categories = os.getenv(
    "BAD_CATEGORIES", "screenshot,blurry,low resolution,low quality")
BAD_CATEGORIES = set(
    cat.strip().lower() for cat in raw_bad_categories.split(",") if cat.strip()
)

# 🧠 Klient
client = Client(host=OLLAMA_HOST)

# Frågeschemat Clef svarar på. Nycklarna matchar BAD_CATEGORIES ovan.
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

# Mappnamn som skrivs direkt efter Clef's kategorisvar.
CLEF_MOVE_MAP = {
    "screenshot": "screenshots",
    "photo_of_screen": "screenshots",
    "low_resolution": BAD_QUALITY_FOLDER_NAME,
    "document": BAD_QUALITY_FOLDER_NAME,
    "blurry": BAD_QUALITY_FOLDER_NAME,
}

if CLASSIFIER not in ("chat", "clef"):
    sys.exit(f"CLASSIFIER måste vara 'chat' eller 'clef', inte {CLASSIFIER!r}")

print(f"🔄 Sorting images... (classifier: {CLASSIFIER})")

# Create folders
os.makedirs(BAD_FOLDER, exist_ok=True)
os.makedirs(OK_FOLDER, exist_ok=True)
if CLASSIFIER == "clef":
    os.makedirs(os.path.join(INPUT_FOLDER, "screenshots"), exist_ok=True)


def classify_image(image_path: str) -> str:
    """Chat-lane: fritextsvar från en vanlig modell."""
    print(f"📸 Classifying image: {image_path}")
    img = Image(value=Path(image_path))  # Use Path directly

    prompt = (
        "You are an image quality classifier. Look extra for screenshots and blurry images.\n"
        "Analyze the image and return ONLY a comma separated list of categories"
    )

    response = client.chat(
        model=MODEL,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [img]
        }]
    )
    return response['message']['content'].strip().lower()


def _encode_image(image_path: str) -> str:
    """Clef tar PNG/JPEG/WebP, max 4 MiB och 16 Mpx per bild. Skala ned först."""
    if PILImage is None:
        with open(image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    with PILImage.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > MAX_PX:
            scale = MAX_PX / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), RESAMPLE)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def classify_image_clef(image_path: str) -> set:
    """Clef-lane: typat schema via /v1/systemone. Returnerar kategorier."""
    print(f"📸 Classifying image (clef): {image_path}")
    body = {
        "model": CLEF_MODEL,
        "state": "Classify this image.",
        "images": [_encode_image(image_path)],
        "questions": CLEF_QUESTIONS,
        "keep_alive": os.getenv("KEEP_ALIVE", "10m"),
    }
    r = requests.post(
        OLLAMA_HOST.rstrip("/") + "/v1/systemone",
        json=body,
        timeout=int(os.getenv("CLEF_TIMEOUT", "180")),
    )
    r.raise_for_status()
    answers = r.json().get("answers", {})

    categories = set()
    itype = answers.get("image_type", {}).get("choice")
    if itype and itype != "photo":
        categories.add(itype)
    blur = answers.get("blurry", {}).get("noul") or 0.0
    if blur >= float(os.getenv("BLUR_THRESHOLD", "0.5")):
        categories.add("blurry")
    return categories


def get_unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    i = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{i}{ext}"
        i += 1
    return new_path


def is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def pick_folder(categories: set) -> str:
    """Översätt kategori-mängd till mål­mapp."""
    if CLASSIFIER == "clef":
        for cat in ("screenshot", "photo_of_screen", "low_resolution", "document", "blurry"):
            if cat in categories:
                name = CLEF_MOVE_MAP.get(cat, BAD_QUALITY_FOLDER_NAME)
                return os.path.join(INPUT_FOLDER, name)
        return OK_FOLDER
    if any(category in categories for category in BAD_CATEGORIES):
        return BAD_FOLDER
    return OK_FOLDER


def sort_images():
    for filename in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, filename)

        print(f"🔍 Checking {filename}...")

        # Skip non-image files or directories
        if not os.path.isfile(file_path):
            continue
        if not is_supported_image(filename):
            continue

        print(f"🔍 Processing {filename}...")
        try:
            if CLASSIFIER == "clef":
                categories = classify_image_clef(file_path)
            else:
                categories = classify_image(file_path)
            print(f"✅ Classified as: {categories}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

        target_folder = pick_folder(categories if isinstance(categories, set)
                                    else {c.strip() for c in str(categories).split(",")})

        destination_path = get_unique_path(
            os.path.join(target_folder, filename))

        shutil.move(file_path, destination_path)


if __name__ == "__main__":
    sort_images()
