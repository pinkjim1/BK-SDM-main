import torch
import torch.nn as nn
import torch.optim as optim
import transformers.models.clip.modeling_clip as clip_modeling
from transformers import CLIPTokenizer, AutoProcessor, AutoTokenizer,CLIPTextConfig
from CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from transformers.models.clip.modeling_clip import CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from read_data import CustomImageDataset, CustomDataLoader
from SupConLoss import SupConLoss
from PIL import Image

text_label=['tench','goldfish', 'great white shark','tiger shark','hammerhead', 'electric ray','stingray', 'cock','hen', 'ostrich']
folder_list=['n01440764','n01443537', 'n01484850', 'n01491361','n01494475','n01496331', 'n01498041','n01514668','n01514859','n01518878']

clip_modeling.CLIPTextEmbeddings=CustomCLIPTextEmbeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config=CLIPTextConfig.from_pretrained("../CLIP")
text_encoder = CLIPTextModelWithProjection.from_pretrained("../CLIP")
image_encoder=CLIPVisionModelWithProjection.from_pretrained("../CLIP")
processor = AutoProcessor.from_pretrained("../CLIP")
tokenizer = CLIPTokenizer.from_pretrained("../CLIP")

text_encoder.to(device)
image_encoder.to(device)

# 冻结 image_encoder 的所有参数
for param in image_encoder.parameters():
    param.requires_grad = False

# 冻结 text_encoder 的所有参数
for param in text_encoder.parameters():
    param.requires_grad = False


text_inputs=tokenizer(text_label, padding=True, return_tensors="pt")
pretrained_embeddings = text_encoder.text_model.embeddings.token_embedding.weight
vt=VirtualTokenManager(text_inputs["input_ids"], pretrained_embeddings).to(device)

optimizer = optim.AdamW(vt.virtual_tokens.parameters(), lr=1e-4)
text_encoder.text_model.embeddings.virtual_tokens=vt

vt.train()

dataset=CustomImageDataset()

dataloader = CustomDataLoader(dataset, batch_size=64, shuffle=True, preprocess=processor, device=device)

loss_fn=SupConLoss(device)

for batch_images, batch_labels in dataloader:
    text_feature=text_encoder(**batch_labels)
    image_feature=image_encoder(**batch_images)
    ll=['_'.join(map(str, row.tolist())) for row in batch_labels['input_ids'][:, 5:-1]]
    loss_t = loss_fn(text_feature['text_embeds'], image_feature['image_embeds'], ll,ll)
    loss_i = loss_fn(image_feature['image_embeds'], text_feature['text_embeds'], ll,ll)
    loss=loss_t+loss_i
    optimizer.zero_grad()  # 清空上一步的梯度
    loss.backward()  # 计算梯度
    optimizer.step()
    print(loss.item())

torch.save(vt.state_dict(), 'vt_checkpoint.pth')
