import torch
import torch.nn as nn
from typing import Optional
from transformers.models.clip.modeling_clip import CLIPTextConfig, CLIPTextEmbeddings

#virtual token
class VirtualTokenManager(nn.Module):
    def __init__(self, token_dim, categories, config:CLIPTextConfig):
        super(VirtualTokenManager, self).__init__()
        self.emb=nn.Embedding(config.vocab_size, token_dim)
        self.virtual_tokens = nn.ParameterDict({str(category[1].item()):self.emb(category[1]) for category in categories})


    def forward(self, categories):
        batch_tokens = []
        for category in categories:
            batch_tokens.append(self.virtual_tokens[str(category)])
        # 返回所有虚拟 token 作为一个 batch
        return torch.stack(batch_tokens)





# combine prompt embedding and virtual token
class CustomCLIPTextEmbeddings(CLIPTextEmbeddings):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.virtual_tokens = None

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        categories = input_ids[:, 5].tolist()
        virtual_token_embeds=self.virtual_tokens(categories)
        inputs_embeds[:, 5, :]=self.virtual_tokens(categories)
        print(virtual_token_embeds.shape)  # 确认形状是否正确
        print(inputs_embeds[:, 5, :].shape)


        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings

