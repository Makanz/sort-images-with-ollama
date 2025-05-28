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
            target_folder = BAD_FOLDER
        else:
            target_folder = OK_FOLDER

        destination_path = os.path.join(target_folder, filename)

        if os.path.exists(destination_path):
            base, ext = os.path.splitext(filename)
            i = 1
            new_filename = f"{base}_{i}{ext}"
            new_destination_path = os.path.join(target_folder, new_filename)
            while os.path.exists(new_destination_path):
                i += 1
                new_filename = f"{base}_{i}{ext}"
                new_destination_path = os.path.join(
                    target_folder, new_filename)
            destination_path = new_destination_path

        shutil.move(file_path, destination_path)


if __name__ == "__main__":
    sort_images()
