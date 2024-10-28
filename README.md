总体介绍：

mid_test.py为总体运行文件，所有的client的内容在p_tuning文件夹中。如果需要修改训练参数可以直接修改p_tuning中的config.yaml文件

本实验需要两个模型，在model文件夹下创建两个文件夹“CLIP”和“bk-sdm-v2-small”。

dataset参考 https://github.com/TsingZ0/PFLlib 生成数据


总体代码流程：
创建client-》测试正确率-》训练prompt-》训练image encoder-》再测试正确率

2-4步循环···

测试结果在根目录的results.csv下

目前模型的缺陷：

1 只能进行cifar10 cifar100的测试和训练（读取的问题，这个应该很快能解决）
2 prompt learning的模型是clip，不是sd（略难）
3 cluster还是普通的cluster（略难）
4 流程固定（其实单独写一个测试也ok，但是我感觉还是有点麻烦，还不如自动测试自动记录）


有些奇葩的地方：
在不独立同分布的client上，因为有些client可能只有两个图片种类，然后正确率就可能会特别高，loss可能会是0.。。