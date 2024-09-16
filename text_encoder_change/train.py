import torch
import os
import torch.nn as nn
import torch.optim as optim
import mobileclip
import pickle
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader


# 生成dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成 512 维度嵌入的 MobileCLIP 模型
mobile_tokenizer = mobileclip.get_tokenizer('mobileclip_s2')
mobile_model, _, _ = mobileclip.create_model_and_transforms('mobileclip_s2', pretrained='MobileCLIP-S2/mobileclip_s2.pt')
mobile_model.to(device)

# 生成 1024 维度嵌入的 CLIPTextModel 模型
clip_tokenizer = CLIPTokenizer.from_pretrained("../bk-sdm-v2-small/tokenizer")
clip_text_encoder = CLIPTextModel.from_pretrained("../bk-sdm-v2-small/text_encoder")
clip_text_encoder.to(device)


# 映射网络，将 512 维度扩展到 1024 维度
class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.bn1 = nn.BatchNorm1d(512)
        self.layer1 = nn.Linear(512, 768)
        self.layer2 = nn.Linear(768, 896)
        self.layer3 = nn.Linear(896, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=x.permute(0,2,1)
        x=self.bn1(x)
        x=x.permute(0,2,1)
        x=self.layer1(x)
        x=self.relu(self.layer2(x))
        x = self.layer3(x)
        x = x.permute(0, 2, 1)
        x = self.bn3(x)
        x = x.permute(0, 2, 1)
        return x


# 初始化映射网络
mapping_network = MappingNetwork().to(device)


mse_loss = nn.MSELoss() # 使用 mse
optimizer = optim.Adam(mapping_network.parameters(), lr=0.001)

with open('train_rest.pkl', 'rb') as f:
    data_list = pickle.load(f)

dataset=CustomDataset(data_list)
dataloader=DataLoader(dataset,batch_size=32,shuffle=True)

# 初始化文件路径
checkpoint_dir = '../checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# 数据加载器，可以提供输入文本对
for epoch in range(50):
    mapping_network.train()
    for inputs in dataloader:  # inputs 包含输入数据
        # 从 MobileCLIP 生成 512 维度嵌入
        mobile_text_input = mobile_tokenizer(inputs)
        mobile_text_input=mobile_text_input.to(device)
        mobile_text_embeddings = mobile_model.encode_text(mobile_text_input)  # (batch_size, 512)

        # 通过映射网络将其扩展为 1024 维度
        expanded_embeddings = mapping_network(mobile_text_embeddings)  # (batch_size, 1024)

        # 从 CLIPTextModel 生成 1024 维度嵌入
        clip_text_input = clip_tokenizer(inputs, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True,
                               return_tensors="pt")

        clip_text_embeddings = clip_text_encoder(clip_text_input.input_ids.to(device))[0]


        # 计算损失
        loss = mse_loss(expanded_embeddings, clip_text_embeddings)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch + 1}.pth')
    torch.save(mapping_network, checkpoint_path)
    print(f'Model checkpoint saved at epoch {epoch + 1}')
    print(f'Epoch [{epoch + 1}/{50}], Loss: {loss.item()}')

