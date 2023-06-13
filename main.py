from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL
from diffusers import LMSDiscreteScheduler
from diffusers import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from PIL import Image
from src.image_defusion_model import ImageDiffusionModel

device = 'cpu'

# Load autoencoder
vae = AutoencoderKL.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='vae').to(device)
# Load tokenizer and the text encoder
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
# Load UNet model
unet = UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4', subfolder='unet').to(device)
# Load schedulers
scheduler_LMS = LMSDiscreteScheduler(beta_start=0.00085,
                                     beta_end=0.012,
                                     beta_schedule='scaled_linear',
                                     num_train_timesteps=1000)

scheduler_DDIM = DDIMScheduler(beta_start=0.00085,
                               beta_end=0.012,
                               beta_schedule='scaled_linear',
                               num_train_timesteps=1000)

model = ImageDiffusionModel(vae, tokenizer, text_encoder, unet, scheduler_LMS, scheduler_DDIM)

while True:
    prompt = input("Enter a prompt (or 'q' to quit): ")
    if prompt == 'q':
        break

    imgs = model.prompt_to_img(prompt)
    imgs[0].save('/tmp/stable-diffusion/test.png')  # Save the image
