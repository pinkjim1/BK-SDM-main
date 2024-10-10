import numpy as np
import os
import torch


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

# 列出文件中存储的数组名称
train_file="0.npz"
with open(train_file, 'rb') as f:
    train_data = np.load(f, allow_pickle=True)['data'].tolist()

X_train = torch.Tensor(train_data['x']).type(torch.float32)
y_train = torch.Tensor(train_data['y']).type(torch.int64)
print("i am here")



images=0
Image=0
# 查看提取的数据
images = images.reshape(-1, 3, 32, 32)  # 先reshape成 (10000, 3, 32, 32)
images = images.transpose(0, 2, 3, 1)  # 再转换成 (10000, 32, 32, 3)，方便后续处理

print("Reshaped Images shape:", images.shape)

image_index = 1  # 你可以更改这个索引来保存不同的图片
image = images[image_index]

# 将 NumPy 数组转换为 PIL 图像对象
image_pil = Image.fromarray(image)
#
# 保存图片为 PNG 格式
image_pil.save(f'cifar_image_{image_index}.png')

print(f"Image cifar_image_{image_index}.png has been saved successfully!")



# 打印图像和标签的形状
