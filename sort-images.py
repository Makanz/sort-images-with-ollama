"""Sort images with a local Ollama model.

Two lanes, chosen with CLASSIFIER:

  chat   (default)  an ordinary chat model replies with a comma separated list
                    of categories. Simple, works with any model.
  clef              a decision model answers a typed schema in one pass via
                    Ollama's /v1/systemone endpoint (Ollama >= 0.35.1).
                    Returns calibrated probabilities instead of free text.

The clef lane also reports how sure it is. When it is unsure, set
SECOND_OPINION to have a different model look at the same image and let the
two verdicts decide together.

    pip install -r requirements.txt
    cp .env.example .env
    python sort-images.py
"""
import base64
import io
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SCREENSHOT_FOLDER_NAME = os.getenv('SCREENSHOT_FOLDER_NAME', 'screenshots')
# Parallella anrop. Ger bara något om servern kör OLLAMA_NUM_PARALLEL >= WORKERS
# OCH modellen lämnar VRAM över till flera samtidiga slots — annars köar bara
# anropen hos Ollama. Se README: "Parallel workers".
WORKERS = max(1, int(os.getenv('WORKERS', '1')))
MAX_PX = int(os.getenv('CLEF_MAX_PX', '1280'))
BLUR_THRESHOLD = float(os.getenv('BLUR_THRESHOLD', '0.5'))

# ── Second opinion ────────────────────────────────────────────────────────
# När Clef är osäker får en annan modell titta på samma bild. Tom = av.
SECOND_OPINION = os.getenv('SECOND_OPINION', '').strip()
SECOND_MODEL = os.getenv('SECOND_OPINION_MODEL', 'gemma3:4b')
SECOND_THRESHOLD = float(os.getenv('SECOND_OPINION_THRESHOLD', '0.8'))
SECOND_MODE = os.getenv('SECOND_OPINION_MODE', 'veto').strip().lower()

BAD_FOLDER = os.path.join(INPUT_FOLDER, BAD_QUALITY_FOLDER_NAME)
OK_FOLDER = os.path.join(INPUT_FOLDER, OK_QUALITY_FOLDER_NAME)
SCREENSHOT_FOLDER = os.path.join(INPUT_FOLDER, SCREENSHOT_FOLDER_NAME)

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

# 📸 chat-lanen: kategorier som ska till skärmdumpsmappen i stället för
# bad_quality. Läses före BAD_CATEGORIES eftersom "screenshot" står i båda.
raw_screenshot_categories = os.getenv(
    "SCREENSHOT_CATEGORIES", "screenshot,photo of screen")
SCREENSHOT_CATEGORIES = set(
    cat.strip().lower() for cat in raw_screenshot_categories.split(",")
    if cat.strip()
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

# Kategorier som betyder "den här bilden ska bort". OBS: BAD_CATEGORIES är
# fritext från chat-lanen och stavar "low resolution" med blanksteg, medan
# clef-lanen använder "low_resolution" — jämför därför mot normaliserade former.
CLEF_BAD = set(CLEF_MOVE_MAP)
BAD_NORMALIZED = {c.replace(" ", "_") for c in BAD_CATEGORIES}

if CLASSIFIER not in ("chat", "clef"):
    sys.exit(f"CLASSIFIER måste vara 'chat' eller 'clef', inte {CLASSIFIER!r}")

print(f"🔄 Sorting images... (classifier: {CLASSIFIER})")

# Create folders
os.makedirs(BAD_FOLDER, exist_ok=True)
os.makedirs(OK_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)


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


def classify_image_clef(image_path: str) -> tuple[set, float | None]:
    """Clef-lane: typat schema via /v1/systemone. Returnerar (kategorier, confidence)."""
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
    if blur >= BLUR_THRESHOLD:
        categories.add("blurry")

    # Clef lämnar kalibrerade sannolikheter. confidence är hur koncentrerad
    # fördelningen är — inte sannolikheten att svaret är rätt. Låg confidence
    # betyder att modellen tvekar mellan flera alternativ.
    conf = answers.get("image_type", {}).get("confidence")
    return categories, conf


def second_opinion(image_path: str) -> set:
    """Låt en annan modell svara på samma bild. Returnerar dess kategorier."""
    print(f"🧐 Second opinion ({SECOND_MODEL}): {image_path}")
    img = Image(value=Path(image_path))
    prompt = (
        "You are an image quality classifier. Look extra for screenshots and blurry images.\n"
        "Analyze the image and return ONLY a comma separated list of categories"
    )
    response = client.chat(
        model=SECOND_MODEL,
        messages=[{'role': 'user', 'content': prompt, 'images': [img]}],
    )
    text = response['message']['content'].strip().lower()
    return {c.strip() for c in text.split(",") if c.strip()}


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
    # chat-lanen får samma tre hinkar: en skärmdump är skräp, men den ska inte
    # blandas ihop med suddiga foton. Kontrollera före BAD_CATEGORIES — modellen
    # kan svara "screenshot", och den kategorin står i båda listorna.
    if any(category in categories for category in SCREENSHOT_CATEGORIES):
        return SCREENSHOT_FOLDER
    if any(category in categories for category in BAD_CATEGORIES):
        return BAD_FOLDER
    return OK_FOLDER


def resolve(categories: set, confidence: float | None,
            image_path: str) -> tuple[set, str]:
    """Clef först; vid tvekan får en andra modell väga in.

    Returnerar (kategorier, spår) där spåret är avsett för loggen så att det
    går att se i efterhand hur många bilder som faktiskt eskalerade.
    """
    if CLASSIFIER != "clef" or not SECOND_OPINION:
        return categories, "clef"
    if confidence is not None and confidence >= SECOND_THRESHOLD:
        return categories, "clef"

    try:
        alt = second_opinion(image_path)
    except Exception as e:
        print(f"⚠️ Second opinion misslyckades ({e}) — behåller Clef")
        return categories, "clef"

    # Andra modellen svarar i sin egen vokabulär. Normalisera till gemensamma
    # kategorinamn så att jämförelsen mot CLEF_BAD/MOVEMAP blir rättvisande.
    alt = {c.replace(" ", "_") for c in alt}

    if SECOND_MODE == "override":
        print(f"🔀 Second opinion ersätter: Clef={categories or '{}'} → {alt or '{}'}")
        return alt, "second-override"

    if SECOND_MODE == "agree":
        # Konservativt: fäll bara om BÅDA flaggar problem. Är de oense vinner
        # tvivlet och bilden behålls — det är hela poängen med läget.
        clef_bad = bool(categories & CLEF_BAD)
        alt_bad = bool(alt & CLEF_BAD)
        if clef_bad and alt_bad:
            print(f"🔀 Båda ense om problem: Clef={categories} andra={alt}")
            return categories, "second-agree"
        if clef_bad or alt_bad:
            print(f"🔀 Oense (Clef={categories or 'ok'} andra={alt or 'ok'}) "
                  f"— behåller bilden")
        return set(), "second-agree-keep"

    # veto (standard): andra modellen får bara rädda bilder, aldrig fälla dem.
    # Clef är den säkrare av de två, så dess dom står kvar om den flaggat.
    if not categories and (alt & CLEF_BAD):
        print(f"🔀 Second opinion räddade bilden: {alt}")
        return alt, "second-veto"
    return categories, "clef"


def classify_and_log(filename: str) -> tuple[set, str | None] | None:
    """Klassificera en bild i INPUT_FOLDER. None = hoppa över (fel)."""
    file_path = os.path.join(INPUT_FOLDER, filename)
    print(f"🔍 Processing {filename}...")
    try:
        if CLASSIFIER == "clef":
            categories, confidence = classify_image_clef(file_path)
            categories, lane = resolve(categories, confidence, file_path)
            print(f"✅ Classified as: {categories or '{}'}  "
                  f"(confidence={confidence}, spår={lane})")
            return categories, lane
        answer = classify_image(file_path)
        print(f"✅ Classified as: {answer}")
        return {c.strip() for c in str(answer).split(",") if c.strip()}, None
    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")
        return None


def sort_images():
    filenames = sorted(
        name for name in os.listdir(INPUT_FOLDER)
        if os.path.isfile(os.path.join(INPUT_FOLDER, name))
        and is_supported_image(name)
    )
    if not filenames:
        print("📂 Inga bilder att sortera.")
        return
    if WORKERS > 1:
        print(f"🧵 {WORKERS} parallella anrop (kräver OLLAMA_NUM_PARALLEL>={WORKERS})")

    # Klassificeringen är det som tar tid och går att köra parallellt. Flytten
    # sker i huvudtråden, så fort just den bilden är klar: get_unique_path()
    # kollar existens och är inte atomär, och en flytt per bild gör att mappen
    # fylls under körningen — avbryter man halvvägs är arbetet inte förlorat.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(classify_and_log, name): name for name in filenames}
        moved = 0
        for future in as_completed(futures):
            filename = futures[future]
            try:
                result = future.result()
            except Exception as e:  # oväntat fel utanför klassificerarens egen try
                print(f"⚠️ Error processing {filename}: {e}")
                continue
            if result is None:
                continue
            categories, _lane = result
            destination_path = get_unique_path(
                os.path.join(pick_folder(categories), filename))
            shutil.move(os.path.join(INPUT_FOLDER, filename), destination_path)
            moved += 1
    print(f"✅ Sorterade {moved} av {len(filenames)} bilder.")


if __name__ == "__main__":
    sort_images()
