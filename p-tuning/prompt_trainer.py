import torch
import torch.nn as nn
import torch.optim as optim
import transformers.models.clip.modeling_clip as clip_modeling
from transformers import CLIPTokenizer, AutoProcessor, AutoTokenizer,CLIPTextConfig
from CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPVisionModel
from PIL import Image

clip_modeling.CLIPTextEmbeddings=CustomCLIPTextEmbeddings


config=CLIPTextConfig.from_pretrained("../CLIP")
text_encoder = CLIPTextModel.from_pretrained("../CLIP")
image_encoder=CLIPVisionModel.from_pretrained("../CLIP")
processor = AutoProcessor.from_pretrained("../CLIP")
tokenizer = CLIPTokenizer.from_pretrained("../CLIP")

text=["cat", "dog"]
text_inputs=tokenizer(text, padding=True, return_tensors="pt")
vt=VirtualTokenManager(1024, text_inputs["input_ids"], config)

optimizer = optim.AdamW(vt.virtual_tokens.parameters(), lr=1e-4)
text_encoder.text_model.embeddings.virtual_tokens=vt

text=["a photo of a cat", "a photo of a dog"]
text_inputs=tokenizer(text, padding=True, return_tensors="pt")
tt=text_encoder(text_inputs.input_ids)






text=["a photo of a cat", "a photo of a dog"]
a=2

text_inputs=AutoTokenizer(text, padding=True, return_tensors="pt")

text_features = model.get_text_features(**text_inputs)