
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler
)

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL, torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=torch.float16
)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_image(prompt, input_image, steps, guidance):
    if input_image is None:
        raise gr.Error("Please upload a reference image.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    image = input_image.convert("RGB").resize((512, 512))
    edges = cv2.Canny(np.array(image), 100, 200)
    control = Image.fromarray(np.stack([edges] * 3, axis=-1))

    result = pipe(
        prompt=prompt,
        image=control,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        controlnet_conditioning_scale=0.85
    ).images[0]

    return control, result

with gr.Blocks(theme=gr.themes.Soft(), title="ControlAI Studio") as demo:
    gr.Markdown("# ControlAI Studio")
    gr.Markdown("AI Image Generation using Stable Diffusion and ControlNet")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Reference Image", type="pil")
            prompt = gr.Textbox(label="Enter Prompt", lines=3)
            steps = gr.Slider(10, 30, value=15, step=1, label="Inference Steps")
            guidance = gr.Slider(1, 12, value=7.5, step=0.5, label="Guidance Scale")
            button = gr.Button("Generate AI Image", variant="primary")

        with gr.Column():
            control_output = gr.Image(label="ControlNet Edge Map")
            output = gr.Image(label="Generated AI Image")

    button.click(
        generate_image,
        inputs=[prompt, input_image, steps, guidance],
        outputs=[control_output, output]
    )

demo.launch()
