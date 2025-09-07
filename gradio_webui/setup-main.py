#########!pip install diffusers transformers accelerate safetensors

import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
lora_model = "abdabd22001/micheal_scott_LoRA"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLPipeline.from_pretrained(base_model, torch_dtype=dtype)
pipe.to(device)
pipe.load_lora_weights(lora_model)

LOGO_PATH = "webui_pictures/dunder-mifflin-featured-image.png"
GIF_PATH = "webui_pictures/dunder-mifflin-this-is-pam.gif"

def generate_images(prompt, n, resolution):
    yield "Loading... Please wait!", [GIF_PATH]
    response_text = f"Prompt: {prompt}\nImages: {n}\nResolution: {resolution}\n"
    images = []

    width, height = map(int, resolution.split("x"))

    for i in range(n):
        image = pipe(prompt, height=height, width=width, num_inference_steps=25).images[0]
        images.append(image)

    yield response_text, images

css = """
#office_logo img {
    width: 300px !important;
    height: 350px !important;
}

#generate-btn {
    font-size: 18px !important;
    padding: 5px 0px !important;
    border-radius: 5px !important;
    color: white !important;
    background-color: #0072ce !important;
    border: none !important;
    cursor: pointer;
    transition: background-color 0.3s;
}
#generate-btn:hover { background-color: #005ea1 !important; }

#center-row { display: flex; justify-content: center; margin: 0px 0; }

#header-bar { margin-bottom: 0px !important; padding-bottom: 0px !important; }

#header-text {
    margin-bottom: -15px !important;
    padding-bottom: 0px !important;
    height: auto !important;
}
#header-text p { font-size: 17px !important; }
#header-text h2 { font-size: 35px !important; font-weight: 800 !important; color: #2c3e50 !important; }
"""

with gr.Blocks(css=css) as app:

    # --- Top row: 
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(
                """
                <h2 style="font-size: 36px; font-weight: bold; margin-bottom: 10px;">
                    Welcome to Dunder Mifflin’s Creative Desk 🏢
                </h2>
                <p>World’s Best Boss meets World’s Best AI.</p>
                <p>Bring Michael Scott in <i>The Office</i> into your AI art.</p>
                <p>Type a creative prompt, pick the number of images, choose resolution, and click Generate.
                <b>That’s what she said.</b></p>
                """,
                elem_id="header-text"
            )
        with gr.Column(scale=1):
            gr.Image(LOGO_PATH, interactive=False, show_label=False, show_download_button=False, container=False)

    # --- Input row ---
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...(A photo of Micheal Scott from The Office ...)")
        n = gr.Slider(minimum=1, maximum=5, step=1, label="Images Requested")
        resolution = gr.Radio(choices=["512x512", "768x768", "1024x1024"], label="Resolution", value="1024x1024")

    with gr.Row(elem_id="center-row"):
        generate_button = gr.Button("Generate", elem_id="generate-btn", scale=0)

    # --- Output row ---
    with gr.Row():
        result_text = gr.Textbox(label="Output", lines=5, interactive=False)
        images_output = gr.Gallery(label="Generated Image")  

    generate_button.click(
        generate_images,
        inputs=[prompt, n, resolution],
        outputs=[result_text, images_output]
    )

app.launch(debug=True, share=True)
