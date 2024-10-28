import torch
import peft
from transformers import CLIPVisionModel
from peft import LoraConfig, PeftModel, get_peft_model,

clip_model_path="../model/CLIP"

clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_path)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['k_proj', 'v_proj', 'q_proj'],
)

new_vision_model = get_peft_model(clip_vision_model, lora_config)

optimizer = torch.optim.AdamW(new_vision_model.parameters(), lr=1e-4)

