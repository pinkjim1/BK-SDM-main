from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms




class CustomImageDataset(Dataset):
    def __init__(self, images, labels, filter_labels):

        self.images = images  # 存储所有图片的路径
        self.labels = labels # 存储图片对应的标签（文件夹名）
        self.filter_labels = filter_labels
        self.append_image=[]
        self.append_label=[]
        for i, label in enumerate(self.labels):
            if label==self.filter_labels:
                self.append_label.append(self.labels[i])
                self.append_image.append(self.images[i])


    def __len__(self):
        # 返回数据集中图片的数量
        return len(self.append_image)

    def __getitem__(self, idx):
        # 根据索引返回图像和标签
        image_path = self.append_image[idx]
        label = self.append_label[idx]

        return image_path, label


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=4, shuffle=True, preprocess=None, total_labels=None, device='cpu', **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.preprocess = preprocess
        self.device = device
        self.total_labels = total_labels

    def __iter__(self):
        # 自定义迭代器
        for batch in super().__iter__():
            images_path, labels = batch
            filtered_images = []
            filtered_labels = []
            for image_path, label in zip(images_path, labels):
                image = Image.open(image_path)
                filtered_images.append(image)
                filtered_labels.append(label)

            pil_images = [image.resize((512, 512)) for image in filtered_images]
            return_value=self.preprocess(text=self.total_labels, images=pil_images, return_tensors="pt", padding=True)
            return_value = return_value.to(self.device)
            yield return_value


