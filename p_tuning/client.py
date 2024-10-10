import transformers.models.clip.modeling_clip as clip_modeling
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.nn as nn
import yaml
import torch.optim as optim
import random
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, AutoProcessor, AutoTokenizer,CLIPTextConfig
from .CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from transformers.models.clip.modeling_clip import CLIPTextModel, CLIPModel
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, StableDiffusionPipeline
from .read_cifar_data import CustomImageDataset, CustomDataLoader
from .SupConLoss import SupConLoss
from PIL import Image


class Client:
    def __init__(self, client_id, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        clip_modeling.CLIPTextEmbeddings = CustomCLIPTextEmbeddings
        self.clip_model_path=config['model_path']['clip_model_path']
        self.sd_model_path=config['model_path']['sd_model_path']
        self.save_model_path=config['model_path']['save_model_path']

        self.prompt_model_lr=config['prompt_model']['lr']
        self.prompt_model_batch_size=config['prompt_model']['batch_size']
        self.prompt_model_weight_decay=config['prompt_model']['weight_decay']
        self.prompt_model_num_epochs=config['prompt_model']['num_epochs']
        self.prompt_model_save_freq=config['prompt_model']['save_freq']

        self.num_inference_steps=config['inference_model']['num_inference_steps']
        self.guidance_scale=config['inference_model']['guidance_scale']
        self.image_num=config['inference_model']['image_num']

        self.dataset_type=config['data']['dataset_type']


        # clip and stable-diffusion model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path)
        cliptokenizer = CLIPTokenizer.from_pretrained(self.clip_model_path)

        #dataset
        self.client_id=client_id
        train_data_dir = os.path.join('dataset', self.dataset_type, 'train', str(self.client_id)+'.npz')
        test_data_dir = os.path.join('dataset', self.dataset_type, 'test', str(self.client_id)+'.npz')
        train_image, train_label=self.load_data(train_data_dir, 'train')
        test_image, test_label=self.load_data(test_data_dir, 'test')
        self.train_dataset = CustomImageDataset(train_image, train_label)
        self.test_dataset = CustomImageDataset(test_image, test_label)

        #prompt
        self.train_prompt=list(set(train_label))
        self.text_inputs = cliptokenizer(self.train_prompt, padding=True, return_tensors="pt")["input_ids"].to(self.device)
        self.pretrained_embeddings = clipmodel.text_model.embeddings.token_embedding.weight.to(self.device)
        self.vt = VirtualTokenManager(self.text_inputs, self.pretrained_embeddings).to(self.device)

        #the chained prompt at this round
        self.prompt_index=0




    def load_data(self, address, type):
        with open(address, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        x_train = torch.Tensor(train_data['x']).type(torch.float32).to('cpu')
        y_train = torch.Tensor(train_data['y']).type(torch.int64).to('cpu')
        to_pil = transforms.ToPILImage()
        x_train = (x_train + 1) / 2
        image = [to_pil(x_train[i]) for i in range(len(x_train))]
        data_dir = os.path.join('dataset', self.dataset_type, type, str(self.client_id))
        image_paths=[]
        for i, image in enumerate(image):
            image_path=os.path.join(data_dir, f'image_{i}.png')
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            image_paths.append(image_path)
        if self.dataset_type == 'Cifar10':
            type_list=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            type_list=[
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
        label=[type_list[i.item()] for i in y_train]
        return image_paths, label

# set requires_grad as false except the prompt which will be trained at this round
    def set_requires_grad(self, grad_keys):
        for name, param in self.vt.virtual_tokens.items():
            if name in grad_keys:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def prompt_train(self):
        clipmodel = CLIPModel.from_pretrained(self.clip_model_path).to(self.device)
        clipprocessor = AutoProcessor.from_pretrained(self.clip_model_path)

        su=sum(p.numel() for p in clipmodel.vision_model.parameters())
        print(su)

        for param in clipmodel.parameters():
            param.requires_grad = False

        clipmodel.text_model.embeddings.virtual_tokens = self.vt
        self.prompt_index = (self.prompt_index + 1)%len(self.train_prompt)
        grad_index=self.text_inputs[self.prompt_index]
        tem_arr = []
        for i in grad_index[1:]:
            if i != 49407:
                tem_arr.append(i)
            else:
                break
        grad_keys='_'.join([str(t.item()) for t in tem_arr])
        # self.set_requires_grad(grad_keys)
        optimizer = optim.AdamW(self.vt.virtual_tokens.parameters(), lr=self.prompt_model_lr, weight_decay=self.prompt_model_weight_decay)
        loss_fn = SupConLoss(self.device)
        dataloader = CustomDataLoader(self.train_dataset, batch_size=self.prompt_model_batch_size, shuffle=True, preprocess=clipprocessor,filter_labels=self.train_prompt[self.prompt_index], device=self.device)
        for epoch in range(self.prompt_model_num_epochs):
            for return_value in dataloader:
                logit = clipmodel(**return_value)
                loss = loss_fn(logit.logits_per_image)
                optimizer.zero_grad()
                loss.backward()  # 清空上一步的梯度 # 计算梯度
                optimizer.step()
                print(loss.item())
            print(f"Epoch[{epoch}/{self.prompt_model_num_epochs}], Loss: {loss.item():.4f}")
            if (epoch + 1) % self.prompt_model_save_freq == 0:
                tem_save_path=os.path.join(self.save_model_path, self.dataset_type,str(self.client_id), str(self.prompt_index), self.train_prompt[self.prompt_index], f"checkpoint_{epoch+1}.pth")
                os.makedirs(os.path.dirname(tem_save_path), exist_ok=True)
                torch.save(self.vt.state_dict(), tem_save_path)
        self.emb_message=(self.train_prompt[self.prompt_index], grad_index,grad_keys, nn.Parameter(self.vt.virtual_tokens[grad_keys].clone(), requires_grad=False))

    def inference(self,messages):
        pipeline = StableDiffusionPipeline.from_pretrained(self.sd_model_path).to(self.device)
        category, token_a, token_b, emb=messages
        emb=emb.to(self.device)
        tem_vt = VirtualTokenManager(token_a, self.pretrained_embeddings).to(self.device)
        tem_vt.virtual_tokens[token_b] = emb
        pipeline.text_encoder.text_model.embeddings.virtual_tokens = tem_vt
        prompt=["a photo of a "+category]
        images=[]
        for i in range(self.image_num):
            generator = None
            result = pipeline(prompt, guidance_scale=self.guidance_scale, num_inference_steps=self.num_inference_steps,
                      generator=generator)
            images.append(result.images[0])
        return category, images


    def exchange_message_and_generate(self, other_clients):
        neighbors = random.sample(other_clients, k=2)  # 随机选择2个邻居

        # 获取邻居的模型参数

        for neighbor in neighbors:
            category, images = self.inference(neighbor.emb_message)
            for i, image in enumerate(images):
                tem_save_path = os.path.join("new_image", str(self.client_id),str(neighbor.client_id), f"{i}.png")
                os.makedirs(os.path.dirname(tem_save_path), exist_ok=True)
                image.save(tem_save_path)






