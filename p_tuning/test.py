import yaml
def test_federated_learning(clients, config_file):
    """
    去中心化联邦学习的主要流程。
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    num_rounds=config['prompt_model']['round']

    client=clients[0]
    client.sd_prompt_train()
    client.exchange_message_and_generate(other_clients=[client])

