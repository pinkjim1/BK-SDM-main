from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms




class CustomImageDataset(Dataset):
    def __init__(self, images, labels):

        self.images = images  # 存储所有图片的路径
        self.labels = labels       # 存储图片对应的标签（文件夹名）

    def __len__(self):
        # 返回数据集中图片的数量
        return len(self.images)

    def __getitem__(self, idx):
        # 根据索引返回图像和标签
        image_path = self.images[idx]
        label = self.labels[idx]

        return image_path, label


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True, preprocess=None,device='cpu', filter_labels=None, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.preprocess = preprocess
        self.device = device
        self.filter_labels = filter_labels

    def __iter__(self):
        # 自定义迭代器
        for batch in super().__iter__():
            images_path, labels = batch
            filtered_images = []
            filtered_labels = []
            for image_path, label in zip(images_path, labels):
                if label in self.filter_labels:  # 只保留在 filter_labels 中的标签
                    image = Image.open(image_path)
                    filtered_images.append(image)
                    filtered_labels.append(label)

            # 如果没有符合条件的图片，跳过这个 batch
            if len(filtered_images) == 0:
                continue

            pil_images = [image.resize((512, 512)) for image in filtered_images]
            tem_text=[f"a photo of a {label}" for label in filtered_labels]
            return_value=self.preprocess(text=tem_text, images=pil_images, return_tensors="pt", padding=True)
            return_value = return_value.to(self.device)
            yield return_value


