import torch
import torch.nn as nn
import yaml
from p_tuning.CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from p_tuning.decentralized_federated_training import decentralized_federated_learning
from p_tuning.test import test_federated_learning
from p_tuning.singal_client_test import Client
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs/single_client_test')

config_file="p_tuning/config.yaml"

id=1
client1=Client(id, config_file, writer)

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

client1.model_test(is_trained=False)
round=10
for i in range(round):
    client1.image_encoder_train()
    client1.model_test(is_trained=True)
