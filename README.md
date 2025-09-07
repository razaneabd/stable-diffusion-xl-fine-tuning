# stable-diffusion-xl-fine-tuning
Michael Scott Image Generation with SDXL

This project fine-tunes Stable Diffusion XL (SDXL 1.0) to generate images of Michael Scott from The Office. Using a combination of DreamBooth and LoRA, the model learns to replicate the character’s unique appearance, facial expressions, and style. The fine-tuned weights can be used for creative prompt-based image generation, deployed through an interactive Gradio app.

Repository Contents

Fine-Tuning Notebook: training_fine-tuning_sdxl.ipynb – Implements DreamBooth + LoRA training on Colab, including preprocessing, hyperparameter tuning, and checkpointing.

Deployment Notebook: gradio_inference.ipynb – Launches a Gradio UI for real-time image generation. Users can input custom prompts, choose resolution (512×512, 768×768, or 1024×1024), and request multiple images.

You can find:

The Dataset: A custom dataset of 23 high-resolution images of Michael Scott, paired with detailed captions in metadata.jsonl.
🔗 Michael Scott Dataset on Hugging Face.

Fine-Tuned Weights: The resulting LoRA weights are lightweight and modular, designed to be applied on top of the SDXL base model.
🔗 Michael Scott LoRA Weights on Hugging Face

Features

⚡ DreamBooth + LoRA fine-tuning for efficient adaptation of SDXL.

🖼️ High-quality image generation up to 1024×1024 resolution.

📦 Modular LoRA checkpoints – small, sharable, and easy to integrate.

💻 Interactive Gradio app with logo, GIF loading animation, and user-friendly controls.

Usage

Fine-tuning

Open training_fine-tuning_sdxl.ipynb in Colab.

Upload the dataset or connect it directly from Hugging Face.

Run training with GPU (T4/A100 recommended).

Deployment

Open gradio_inference.ipynb.

Load the fine-tuned LoRA weights with the SDXL base model.

Launch the Gradio app and start generating!
