import torch
from torch import nn
from torch.nn.init import xavier_normal_
from model.SeqRec import ContextEncoder
import numpy as np
from model_utils.BERT_SeqRec import BertModel
from model_utils.BERT_SeqRec import BertConfig


class BERT_encoder(ContextEncoder):
    def __init__(self, config):
    # self.user_emb = nn.Embedding(config['user_num'], config['hidden_size'])
    # self.item_emb = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0)
        super().__init__(config)
        print('------------BERT ENCODER YES------------')
        self.network_param_init(config)
    
    def network_param_init(self, config):  # 设置seq_module的参数
        bert_config = BertConfig(config['item_num'], config)  # bert的词典大小为item_num 其余遵循config
        self.seq_module = BertModel(bert_config, use_outer_embed=True)  # outer_embed为true，其实是

    def forward(self, hist_item_ids, masks):
        bert_context = self.seq_module(hist_item_ids, attention_mask=masks, outer_embed=self.item_emb.weight)
        return bert_context[:, -1, :].squeeze(1)
    
