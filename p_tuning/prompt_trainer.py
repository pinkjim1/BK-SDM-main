import torch
import torch.nn as nn
import torch.optim as optim
import transformers.models.clip.modeling_clip as clip_modeling
from transformers import CLIPTokenizer, AutoProcessor, AutoTokenizer,CLIPTextConfig
from CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPModel
from read_data import CustomImageDataset, CustomDataLoader
from SupConLoss import SupConLoss
from PIL import Image

text_label=['tench','goldfish', 'great white shark','tiger shark','hammerhead', 'electric ray','stingray', 'cock','hen', 'ostrich']
folder_list=['n01440764','n01443537', 'n01484850', 'n01491361','n01494475','n01496331', 'n01498041','n01514668','n01514859','n01518878']

clip_modeling.CLIPTextEmbeddings=CustomCLIPTextEmbeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=CLIPModel.from_pretrained("../CLIP")
processor = AutoProcessor.from_pretrained("../CLIP")
tokenizer = CLIPTokenizer.from_pretrained("../CLIP")


model.to(device)


for param in model.parameters():
    param.requires_grad = False





text_inputs=tokenizer(text_label, padding=True, return_tensors="pt")
pretrained_embeddings = model.text_model.embeddings.token_embedding.weight
vt=VirtualTokenManager(text_inputs["input_ids"], pretrained_embeddings).to(device)

optimizer = optim.AdamW(vt.virtual_tokens.parameters(), lr=1e-4)
model.text_model.embeddings.virtual_tokens=vt

vt.train()

dataset=CustomImageDataset()

dataloader = CustomDataLoader(dataset, batch_size=64, shuffle=True, preprocess=processor, device=device)

loss_fn=SupConLoss(device)

num_epochs = 50
save_interval=10

for epoch in range(num_epochs):
    print(epoch)
    for return_value in dataloader:
        logit=model(**return_value)
        tensor_label=return_value['input_ids'][:,5:7]
        label=["_".join(map(str,row.tolist())) for row in tensor_label]
        loss_i=loss_fn(logit.logits_per_image,label, label)
        loss_t=loss_fn(logit.logits_per_text,label, label)
        loss=loss_i+loss_t
        optimizer.zero_grad()
        loss.backward()# 清空上一步的梯度 # 计算梯度
        optimizer.step()
        print(loss.item())
    print(f"Epoch[{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
    if (epoch+1)%save_interval==0:
        torch.save(vt.state_dict(),f"checkpoint/vt_checkpoint_{epoch+1}.pth")
