import os
import shutil
from pathlib import Path
from ollama import chat
from ollama._types import Image
from ollama import Client
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://host.docker.internal:11434')
client = Client(host=OLLAMA_HOST)

# Paths
INPUT_FOLDER = os.getenv('INPUT_FOLDER', 'images')
BAD_QUALITY_FOLDER_NAME = os.getenv('BAD_QUALITY_FOLDER_NAME', 'bad_quality')
BAD_FOLDER = os.path.join(INPUT_FOLDER, BAD_QUALITY_FOLDER_NAME)
OK_QUALITY_FOLDER_NAME = os.getenv('OK_QUALITY_FOLDER_NAME', 'ok')
OK_FOLDER = os.path.join(INPUT_FOLDER, OK_QUALITY_FOLDER_NAME)

print("🔄 Sorting images...")

# Create folders
os.makedirs(BAD_FOLDER, exist_ok=True)
os.makedirs(OK_FOLDER, exist_ok=True)

# Model name – use a vision-capable one
MODEL = os.getenv('MODEL_NAME', 'gemma3:4b')  # Working great

# Supported image extensions
raw_extensions = os.getenv("SUPPORTED_EXTENSIONS",
                           ".jpg,.jpeg,.png,.bmp,.webp")
SUPPORTED_EXTENSIONS = set(ext.strip().lower()
                           for ext in raw_extensions.split(",") if ext.strip())

# Bad categories for classification
raw_bad_categories = os.getenv(
    "BAD_CATEGORIES", "blurry")
BAD_CATEGORIES = set(cat.strip().lower()
                     for cat in raw_bad_categories.split(",") if cat.strip())


def classify_image(image_path: str) -> str:
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
            categories = classify_image(file_path)
            print(f"✅ Classified as: {categories}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

        if any(category in categories for category in BAD_CATEGORIES):
            target_folder = BAD_FOLDER
        else:
            target_folder = OK_FOLDER

        destination_path = get_unique_path(
            os.path.join(target_folder, filename))

        shutil.move(file_path, destination_path)


if __name__ == "__main__":
    sort_images()
