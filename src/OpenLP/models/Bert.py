import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel, BatchEncoding
# from src.utils import roc_auc_score, mrr_score, ndcg_score

from IPython import embed


class BertForLinkPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, center_input, neighbor_input=None, mask=None, **kwargs):

        node_embeddings = self.bert(input_ids=center_input['input_ids'], attention_mask=center_input['attention_mask'])

        return node_embeddings
