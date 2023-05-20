import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel

from IPython import embed

class BertSageForLinkPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.graph_transform = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.pooling_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

    def aggregation(self):
        raise NotImplementedError()

    def graphsage(self, center_embed, neighbor_embed, neighbor_mask):
        
        if len(neighbor_mask.shape) != 2:
            neighbor_mask = neighbor_mask.view(-1, neighbor_mask.shape[-1])

        neighbor_embed = self.aggregation(neighbor_embed, neighbor_mask)  # B D
        main_embed = torch.cat([center_embed, neighbor_embed], dim=-1)  # B 2D

        main_embed = self.graph_transform(main_embed)
        main_embed = F.relu(main_embed)
        return main_embed

    def forward(self, center_input, neighbor_input=None, mask=None, **kwargs):

        # for infer
        if not neighbor_input:
            neighbor_input = center_input.copy()
            mask = torch.zeros(center_input['input_ids'].shape[0], 1, dtype=torch.long).to(center_input['input_ids'].device)
            # node_embeddings = self.bert(input_ids=center_input['input_ids'], attention_mask=center_input['attention_mask'])
            # return node_embeddings

        # adjust neighbor tensor size
        if len(neighbor_input['input_ids'].shape) == 3:
            neighbor_input = {k: v.view(-1, v.shape[-1]) for k, v in neighbor_input.items()}
        
        B, L = center_input['input_ids'].shape
        D = self.config.hidden_size

        input_ids = torch.cat((center_input['input_ids'], neighbor_input['input_ids']))
        attention_mask = torch.cat((center_input['attention_mask'], neighbor_input['attention_mask']))

        hidden_states = self.bert(input_ids, attention_mask)
        center_last_hidden_states = hidden_states.last_hidden_state[:B][:,0]
        neighbor_last_hidden_states = hidden_states.last_hidden_state[B:][:,0].view(B,-1,D)

        node_embeddings = self.graphsage(center_last_hidden_states, neighbor_last_hidden_states, mask)
        
        return node_embeddings


class BertMaxSageForLinkPredict(BertSageForLinkPredict):
    def __init__(self, config):
        super().__init__(config)

    def aggregation(self, neighbor_embed, neighbor_mask):

        neighbor_embed = F.relu(self.pooling_transform(neighbor_embed))
        neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(2) == 0, 0)
        return torch.max(neighbor_embed, dim=-2)[0]


class BertMeanSageForLinkPredict(BertSageForLinkPredict):
    def __init__(self, config):
        super().__init__(config)

    def aggregation(self, neighbor_embed, neighbor_mask):

        neighbor_embed = F.relu(self.pooling_transform(neighbor_embed))
        neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(2) == 0, 0)
        return torch.mean(neighbor_embed, dim=-2)


class BertSumSageForLinkPredict(BertSageForLinkPredict):
    def __init__(self, config):
        super().__init__(config)

    def aggregation(self, neighbor_embed, neighbor_mask):

        neighbor_embed = F.relu(self.pooling_transform(neighbor_embed))
        neighbor_embed = neighbor_embed.masked_fill(neighbor_mask.unsqueeze(2) == 0, 0)
        return torch.sum(neighbor_embed, dim=-2)
