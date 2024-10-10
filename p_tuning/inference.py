import transformers.models.clip.modeling_clip as clip_modeling
import torch

from tqdm.auto import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer
from CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from PIL import Image

clip_modeling.CLIPTextEmbeddings=CustomCLIPTextEmbeddings

class InferenceModel:
    def __init__(self, model_path, embedding_model_path, device):
        self.device = device
        self.model_path=model_path
        self.vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet"
        )
        self.scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.vt = VirtualTokenManager()
        self.vt.load_from_state_dict(torch.load(embedding_model_path))
        self.text_encoder.text_model.embeddings.virtual_tokens=self.vt

    def generate_image(self, prompt, save_path, seed=None):
        height = 512
        width = 512
        num_inference_steps = 25
        guidance_scale = 7.5
        batch_size = len(prompt)
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]


