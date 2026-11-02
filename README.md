# ControlAI Studio

AI image generation application using Stable Diffusion 1.5 and ControlNet Canny.

## Features
- Upload a reference image
- Enter a text prompt
- Adjust inference steps and guidance scale
- Generate a ControlNet edge map
- Generate AI images

## Run
Install dependencies:

    pip install -r requirements.txt

Run:

    python app.py

## Models
- runwayml/stable-diffusion-v1-5
- lllyasviel/control_v11p_sd15_canny

A compatible GPU is recommended.