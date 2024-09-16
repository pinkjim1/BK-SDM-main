from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import torch
from PIL import Image

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("bk-sdm-v2-small/vae", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("bk-sdm-v2-small/tokenizer")
text_encoder = CLIPTextModel.from_pretrained("bk-sdm-v2-small/text_encoder")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("bk-sdm-v2-small/unet", subfolder="unet")


# 4. load the K-LMS scheduler with some fitting parameters
from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

# 5. move the models to GPU
torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)


# 6. set parameters
prompt = ["an horse"]

height = 512                        # default height of Stable Diffusion
width = 512                         # default width of Stable Diffusion

num_inference_steps = 100           # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

batch_size = len(prompt)


# 7. get the text_embeddings for the passed prompt
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
tem=text_input.input_ids
tt=text_encoder(text_input.input_ids.to(torch_device))
text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]




# 8. get the unconditional text embeddings for classifier-free guidance
# They need to have the same shape as the conditional text_embeddings (batch_size and seq_length)
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]


# 9. concatenate both text_embeddings and uncond_embeddings into a single batch to avoid doing two forward passes
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


# 10. generate the initial random noise
latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

# 11. initialize the scheduler with our chosen num_inference_steps.
scheduler.set_timesteps(num_inference_steps)

# 12. The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here:
latents = latents * scheduler.init_noise_sigma


# 13. write the denoising loop
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample


# 14. use the vae to decode the generated latents back into the image
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample

# 15. convert the image to PIL so we can display or save it
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = Image.fromarray(images[0])
print(pil_images)
pil_images.save("example.png")
print(pil_images)


