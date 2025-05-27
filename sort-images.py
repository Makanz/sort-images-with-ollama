import os
import shutil
from pathlib import Path
from ollama import chat
from ollama._types import Image

# Paths
INPUT_FOLDER = 'images'
BAD_FOLDER = os.path.join(INPUT_FOLDER, 'bad_quality')
SCREENSHOT_FOLDER = os.path.join(INPUT_FOLDER, 'screenshots')
OK_FOLDER = os.path.join(INPUT_FOLDER, 'ok')

# Create folders
os.makedirs(BAD_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
os.makedirs(OK_FOLDER, exist_ok=True)

# Model name – use a vision-capable one
MODEL = 'llama3.2-vision'  # or your preferred model, like 'gemma:vision' if available

def classify_image(image_path: str) -> str:
    img = Image(value=Path(image_path))  # Use Path directly
    prompt = (
        "Classify this image as one of the following: screenshot, bad_quality, or ok. "
        "Only respond with one of those three words."
    )

    response = chat(
        model=MODEL,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [img]
        }]
    )
    return response['message']['content'].strip().lower()

def sort_images():
    for filename in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, filename)

        # Skip non-image files or directories
        if not os.path.isfile(file_path):
            continue
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue

        print(f"🔍 Processing {filename}...")
        try:
            category = classify_image(file_path)
            print(f"✅ Classified as: {category}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

        if category == 'screenshot':
            shutil.move(file_path, os.path.join(SCREENSHOT_FOLDER, filename))
        elif category == 'bad_quality':
            shutil.move(file_path, os.path.join(BAD_FOLDER, filename))
        else:
            shutil.move(file_path, os.path.join(OK_FOLDER, filename))

if __name__ == "__main__":
    sort_images()