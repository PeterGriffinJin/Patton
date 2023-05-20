import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel

from IPython import embed

class BertGATForLinkPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.attention_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_weight = nn.Linear(2 * config.hidden_size, 1)
        self.init_weights()

    def aggregation(self, node_embed, node_mask):
        B, CN, D = node_embed.shape

        # node embedding transform
        node_embed = self.attention_transform(node_embed)

        # node mask
        node_mask = torch.cat((torch.ones(B, 1, dtype=torch.long, device=node_mask.device), node_mask), 1)

        # calculate attention score
        center_embed_expand = node_embed[:, :1].expand(B, CN, D)
        attention_score = self.attention_weight(torch.cat((center_embed_expand, node_embed), dim=-1))
        attention_score = F.leaky_relu(attention_score).squeeze(-1)
                
        node_mask_real = (1.0 - node_mask) * -10000.0
        attention_score = attention_score + node_mask_real
        attention_score = F.softmax(attention_score, dim=1).unsqueeze(-1)

        # aggregate
        main_embed = (attention_score * node_embed).sum(dim=1)
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
        
        cls_embeddings = torch.cat((center_last_hidden_states.unsqueeze(1), neighbor_last_hidden_states), 1)
        node_embeddings = self.aggregation(cls_embeddings, mask)

        return node_embeddings
