from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


text_label=['tench','goldfish', 'great white shark','tiger shark','hammerhead', 'electric ray','stingray', 'cock','hen', 'ostrich']
folder_list=['n01440764','n01443537', 'n01484850', 'n01491361','n01494475','n01496331', 'n01498041','n01514668','n01514859','n01518878']

folder=r"C:\Users\79402\.cache\autoencoders\data\ILSVRC2012_train\data"


class CustomImageDataset(Dataset):
    def __init__(self, root_dir=folder):
        """
        Args:
            root_dir (string): 图片文件夹的根目录，包含多个子文件夹，每个子文件夹代表一个类别
            transform (callable, optional): 对图像进行预处理的操作
        """
        self.root_dir = root_dir
        self.image_paths = []  # 存储所有图片的路径
        self.labels = []       # 存储图片对应的标签（文件夹名）

        # 遍历每个子文件夹，获取图片路径和对应的标签
        for label, sub_dir in enumerate(os.listdir(root_dir)):
            if sub_dir not in folder_list:
                continue
            sub_dir_path = os.path.join(root_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                ind=folder_list.index(sub_dir)
                l=text_label[ind]
                for image_name in os.listdir(sub_dir_path):
                    image_path = os.path.join(sub_dir_path, image_name)
                    if image_name.endswith('.JPEG'):
                        self.image_paths.append(image_path)
                        self.labels.append(l)

    def __len__(self):
        # 返回数据集中图片的数量
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 根据索引返回图像和标签
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        return image_path, label


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True, preprocess=None,device='cpu', **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.preprocess = preprocess
        self.device = device

    def __iter__(self):
        # 自定义迭代器
        for batch in super().__iter__():
            images, labels = batch
            pil_images = [Image.open(image_path).resize((512, 512)) for image_path in images]
            tem_text=[f"a photo of a {label}" for label in labels]
            image_inputs=self.preprocess(images=pil_images, return_tensors="pt")
            text_inputs = self.preprocess(text=tem_text, padding=True, return_tensors="pt")
            image_inputs = image_inputs.to(self.device)
            text_inputs = text_inputs.to(self.device)
            yield image_inputs, text_inputs



