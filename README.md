# sort-images-with-ollama/README.md

# Sort Images with Ollama

This project is designed to classify and sort images into different categories using a Python script and the Ollama model. The script processes images located in the `images` directory and sorts them into three categories: screenshots, bad quality, and okay.

## Project Structure

```
sort-images-with-ollama
├── .devcontainer
│   ├── devcontainer.json
│   └── Dockerfile
├── images
├── sort-images.py
└── README.md
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd sort-images-with-ollama
   ```

2. **Open in Development Container**
   - Open the project in Visual Studio Code.
   - Use the command palette (Ctrl+Shift+P) and select `Remote-Containers: Reopen in Container`.

3. **Install Dependencies**
   - The development container will automatically install the required Python dependencies specified in the `Dockerfile`.

## Usage

1. Place the images you want to classify in the `images` directory.
2. Run the script:
   ```bash
   python sort-images.py
   ```
3. The images will be sorted into the following folders:
   - `images/screenshots` for screenshots
   - `images/bad_quality` for images classified as bad quality
   - `images/ok` for images classified as okay

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.