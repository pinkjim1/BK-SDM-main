import pickle
from PIL import Image
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# 读取 npz 文件
data_dict = unpickle("data_batch_1")

# 列出文件中存储的数组名称
images = data_dict[b'data']  # 图片数据
labels = data_dict[b'labels']

print("Images shape:", images.shape)  # CIFAR-10包含10000张图片，每张图片是3072维的扁平数组
print("Labels shape:", len(labels))

# 查看提取的数据
images = images.reshape(-1, 3, 32, 32)  # 先reshape成 (10000, 3, 32, 32)
images = images.transpose(0, 2, 3, 1)  # 再转换成 (10000, 32, 32, 3)，方便后续处理

print("Reshaped Images shape:", images.shape)

image_index = 1  # 你可以更改这个索引来保存不同的图片
image = images[image_index]

# 将 NumPy 数组转换为 PIL 图像对象
image_pil = Image.fromarray(image)

# 保存图片为 PNG 格式
image_pil.save(f'cifar_image_{image_index}.png')

print(f"Image cifar_image_{image_index}.png has been saved successfully!")



# 打印图像和标签的形状
