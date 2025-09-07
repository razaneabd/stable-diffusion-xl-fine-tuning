# Fine-Tuning Stable Diffusion XL on Michael Scott with DreamBooth + LoRA

This repository contains two main components:
1. **Fine-tuning Notebook**: A Google Colab notebook to fine-tune **Stable Diffusion XL (SDXL)** on a custom dataset using **DreamBooth** combined with **LoRA (Low-Rank Adaptation)**.  
2. **Deployment file**: A Gradio-powered interactive WebUI that allows generating images of *Michael Scott* from *The Office* using the fine-tuned LoRA weights.

The project demonstrates how to adapt a large general-purpose text-to-image model (SDXL) to a niche subject by leveraging lightweight fine-tuning techniques.

---

## Overview

Pretrained SDXL models are general-purpose and often fail to reproduce niche or less commonly represented characters. In our case, when prompting the base SDXL model with *“Michael Scott from The Office”*, the generated images did not resemble the character accurately.  

To address this limitation, we applied **DreamBooth fine-tuning with LoRA**:
- **DreamBooth** specializes in personalizing generative models using a small curated dataset.  
- **LoRA** allows efficient fine-tuning by training only low-rank weight matrices instead of updating the entire model, saving compute and storage.  

This combination allowed us to produce more faithful and recognizable generations of Michael Scott while keeping the process lightweight and efficient.

---

## Repository Structure

```plaintext
├── training_fine_tuning_sdxl.ipynb   
├── GradioWebUI/                     
│   ├── setup-main.py                 
│   └── webui_pictures/               
└── README.md                         

## 📂 Resources

The following resources are hosted externally on Hugging Face and can be used with this project:

- **[Michael Scott Dataset](https://huggingface.co/datasets/abdabd22001/micheal_scott)**  
- **[LoRA Weights (Fine-tuned SDXL)](https://huggingface.co/abdabd22001/micheal_scott_LoRA)**  

The Dataset: A custom dataset of 23 high-resolution images of Michael Scott, paired with detailed captions in metadata.jsonl.
🔗 Michael Scott Dataset on Hugging Face.

Fine-Tuned Weights: The resulting LoRA weights are lightweight and modular, designed to be applied on top of the SDXL base model.
🔗 Michael Scott LoRA Weights on Hugging Face



## Requirements

- Google Colab with **T4 GPU** (recommended).  
- Python 3.9+  
- Main libraries:  
  - `diffusers`  
  - `transformers`  
  - `accelerate`  
  - `bitsandbytes`  
  - `peft`  
  - `gradio`  

All dependencies are installed automatically in the notebooks.

## Usage

### 1. Fine-tuning (optional unless you want to fine tune the model)
Open the fine-tuning notebook and run the cells in sequence:  
- Upload the dataset.  
- Resize images (with padding) to `1024x1024`.  
- Set hyperparameters.  
- Launch training.  
- Upload checkpoints to Hugging Face.  

### 2. Deployment with Gradio
Run the Gradio notebook to launch a simple WebUI:  
- Type a prompt describing Michael Scott in *The Office*.  
- Select the number of outputs and resolution.  
- Generate images directly in your browser.  

Note: For best results, use **1024x1024 resolution**. Each generation may take **20–25 seconds**.

---


## Future Work

- Add **quantitative evaluation metrics** (e.g., CLIP similarity scores).  
- Expand dataset to improve variety (different poses, scenes).  
- Extend deployment beyond Gradio to a production-ready web app.  
