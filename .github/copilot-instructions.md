# Instructions for Using LLM in sort-images-with-ollama

## Overview

This project uses a Large Language Model (LLM) to assist with sorting and categorizing images. Follow these steps to set up and use the LLM effectively.

## 1. Prerequisites

- Ensure you have Python 3.8+ installed.
- Install required dependencies using `pip install -r requirements.txt`.
- Download and set up the Ollama LLM as described in the project README.

## 2. Running the LLM

- Start the Ollama LLM server (refer to the Ollama documentation for details).
- Configure the connection settings in the project's configuration file (e.g., `config.yaml` or `.env`).

## 3. Using the LLM for Image Sorting

- Place your images in the designated input folder (see project README).
- Run the main script (e.g., `python sort_images.py`).
- The script will communicate with the LLM to analyze and sort images based on the model's output.

## 4. Customization

- You can modify the prompts sent to the LLM in the code (see `llm_utils.py` or similar files).
- Adjust model parameters as needed for your use case.

## 5. Troubleshooting

- If the LLM is not responding, ensure the server is running and the connection details are correct.
- Check logs for error messages.

## 6. Additional Resources

- See the Ollama documentation for advanced configuration and usage.
- Refer to the project README for more details on image sorting workflows.
