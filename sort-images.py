import os
import shutil
from pathlib import Path
from ollama import chat
from ollama._types import Image
from ollama import Client

client = Client(
    host='http://host.docker.internal:11434'
)

# Paths
INPUT_FOLDER = 'images'
BAD_FOLDER = os.path.join(INPUT_FOLDER, 'bad_quality')
OK_FOLDER = os.path.join(INPUT_FOLDER, 'ok')

print("🔄 Sorting images...")

# Create folders
os.makedirs(BAD_FOLDER, exist_ok=True)
os.makedirs(OK_FOLDER, exist_ok=True)

# Model name – use a vision-capable one
MODEL = 'gemma3:4b'  # Working great

# MODEL = 'llava:latest' # Not working, dosent follow instructions


def classify_image(image_path: str) -> str:
    print(f"📸 Classifying image: {image_path}")
    img = Image(value=Path(image_path))  # Use Path directly

    prompt = (
        "You are an image quality classifier. Look extra for screenshots and blurry images.\n"
        "Analyze the image and return ONLY a comma sperated list of categories"
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


def sort_images():
    for filename in os.listdir(INPUT_FOLDER):
        file_path = os.path.join(INPUT_FOLDER, filename)

        print(f"🔍 Checking {filename}...")

        # Skip non-image files or directories
        if not os.path.isfile(file_path):
            continue
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
            continue

        print(f"🔍 Processing {filename}...")
        try:
            categories = classify_image(file_path)
            print(f"✅ Classified as: {categories}")
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue

        # Check categories and move files accordingly
        if 'screenshot' in categories or 'blurry' in categories or 'low resolution' in categories or 'low quality' in categories:
            shutil.move(file_path, os.path.join(BAD_FOLDER, filename))
        else:
            shutil.move(file_path, os.path.join(OK_FOLDER, filename))


if __name__ == "__main__":
    sort_images()
