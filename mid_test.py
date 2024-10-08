import torch
import torch.nn as nn
import yaml
from p_tuning.CustomCLIPTextEmbeddings import VirtualTokenManager, CustomCLIPTextEmbeddings
from p_tuning.decentralized_federated_training import decentralized_federated_learning
from p_tuning.client import Client
from torch.utils.data import DataLoader, TensorDataset

config_file="p_tuning/config.yaml"

def create_clients(num_clients, config_file):
    clients = []
    for client_id in range(num_clients):
        clients.append(Client(client_id, config_file))
    return clients


with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# 创建客户端
num_clients = config['client']['num']
clients = create_clients(num_clients, config_file)

# 运行去中心化联邦学习
clients = decentralized_federated_learning(clients, config_file)
