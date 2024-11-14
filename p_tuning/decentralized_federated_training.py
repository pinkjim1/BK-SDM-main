import yaml
def decentralized_federated_learning(clients, config_file):
    """
    去中心化联邦学习的主要流程。
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    num_rounds=config['prompt_model']['round']

    for round in range(num_rounds):
        print(f"Round {round+1}/{num_rounds}")

        # 每个客户端在本地训练模型
        for client in clients:
            if round ==0:
                client.model_test(is_trained=False)
            client.prompt_train()

        # 每个客户端与其他客户端交换并聚合模型
        for client in clients:
            client.exchange_message_and_generate([c for c in clients if c != client])
            client.image_encoder_train()
            if round%5==0:
                client.model_test(is_trained=True)

        print("Models exchanged and aggregated.")



