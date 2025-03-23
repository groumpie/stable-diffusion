# 🖼️ Image Diffusion Model

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A powerful and efficient implementation of a text-to-image diffusion model using PyTorch.

## 📝 Description

This repository contains an implementation of a text-to-image diffusion model that generates high-quality images from text prompts. The model uses a combination of a Variational Autoencoder (VAE), text encoder, UNet, and specialized schedulers to progressively transform random noise into coherent images based on textual descriptions.

## ✨ Features

- 🔄 Text-to-image generation via diffusion process
- 🎛️ Configurable image dimensions and generation parameters
- 🧠 Classifier-free guidance for improved image quality
- ⚡ Automatic mixed precision (AMP) for efficient inference
- 🔍 Support for different sampling algorithms (LMS, DDIM)

## 🛠️ Installation

```bash
pip install torch pillow tqdm transformers diffusers
```

## 🚀 Quick Start

```python
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDIMScheduler

# Load pre-trained models
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Initialize our custom diffusion model
diffusion_model = ImageDiffusionModel(
    vae=pipe.vae,
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    unet=pipe.unet,
    scheduler_LMS=LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
    scheduler_DDIM=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
)

# Generate an image
prompt = "A beautiful sunset over mountains with a lake reflection"
images = diffusion_model.prompt_to_img(
    prompts=prompt,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5
)

# Save the result
images[0].save("generated_image.png")
```

## 🧩 Model Architecture

The diffusion model consists of several key components:

- **VAE (Variational Autoencoder)**: Encodes images into latent space and decodes latents back to images
- **Text Encoder**: Converts text prompts into embeddings using a transformer-based architecture
- **UNet**: Predicts noise residuals in the diffusion process
- **Schedulers**: Control the noise addition/removal during the diffusion process
  - LMS (Linear Multistep) Scheduler: Used for the main diffusion process
  - DDIM Scheduler: Alternative sampling method (not used in the basic example)

## 📊 Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `prompts` | Text description(s) for image generation | Required |
| `height` | Height of generated image | 512 |
| `width` | Width of generated image | 512 |
| `num_inference_steps` | Number of denoising steps | 50 |
| `guidance_scale` | Controls adherence to text prompt (higher = more faithful) | 7.5 |
| `img_latents` | Optional initial latent vectors | None |

## 📋 Methods

- `get_text_embeds(text)`: Tokenizes and embeds text using the text encoder
- `get_prompt_embeds(prompt)`: Creates conditional and unconditional embeddings for guidance
- `get_img_latents(...)`: Performs the reverse diffusion process (noise → image)
- `decode_img_latents(img_latents)`: Decodes latent vectors to images using the VAE
- `transform_imgs(imgs)`: Processes images for final output
- `prompt_to_img(...)`: Main method for text-to-image generation

## 🔍 Advanced Usage

### Custom Initial Latents

```python
# Create custom initial latents
latents = torch.randn(1, 4, 64, 64)  # Adjust size based on VAE dimensions

# Generate image using these latents
images = diffusion_model.prompt_to_img(
    prompts="A futuristic city with flying cars",
    img_latents=latents
)
```

### Batch Processing

```python
# Generate multiple images at once
prompts = ["A forest in autumn", "A beach at sunset", "A snowy mountain peak"]
images = diffusion_model.prompt_to_img(prompts=prompts)

# Save all images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")
```

## ⚠️ Limitations

- CPU-only implementation as shown (can be modified for GPU)
- Memory usage increases with image dimensions
- Generation time depends on the number of inference steps

## 🔗 References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

## 📜 License

This project is available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
