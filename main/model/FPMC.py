import torch
from torch import nn
from torch.nn.init import xavier_normal_
from model.SeqRec import ContextEncoder

class FPMCContextEncoder(ContextEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.user_embeddings = nn.Embedding(config['user_num'], config['hidden_size'])
        self.item_embeddings = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0)
        self.UI_emb = nn.Embedding(config['user_num'], config['hidden_size'])
        self.IU_emb = nn.Embedding(config['item_num'], config['hidden_size'])
        self.IL_emb = nn.Embedding(config['item_num'], config['hidden_size'])
        self.LI_emb = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0)
        # self.network_param_init(config)

    def forward(self, user_id, hist_item_ids, target):
        raise NotImplementedError


class FPMC_encoder(FPMCContextEncoder):

    def __init__(self, config):
        super().__init__(config)
        print('------------FPMC ENCODER YES------------')

    def forward(self, user_id, hist_item_ids, target):
        """        print('arg1:', user_id) 
        print('arg2:', hist_item_ids)
        print('arg3',target)"""
        hist_item_embed = self.LI_emb(hist_item_ids)#[b,input_len,emb]

        #print(hist_item_embed)
        '''
        sumed_item_embed = hist_item_embed.sum(dim=1, keepdim=False)
        li_emb = sumed_item_embed
        li_emb= torch.unsqueeze(li_emb, dim=1)  # [b,1,emb]
        '''
        li_emb = hist_item_embed.sum(dim=1, keepdim=True)#[b,1,emb]
        #print(li_emb.size())
        ui_emb = self.UI_emb(user_id)
        #print(user_id.size())
        ui_emb = torch.unsqueeze(ui_emb, dim=1)  # [b,1,emb]
        #print(target)
        iu_emb = self.IU_emb(target)  # [b,1,emb]     
        #print(iu_emb.size())
        il_emb = self.IL_emb(target)  # [b,1,emb]
        #print(il_emb.size())

        return ui_emb, iu_emb, il_emb, li_emb
    
'''  def _get_test_score(self,user_id, hist_item_ids, target_ids):
        user_emb = self.UI_emb(user_id)
        all_iu_emb = self.IU_emb.weight
        mf = torch.matmul(user_emb,all_iu_emb.transpose(0,1))

        hist_item_embed = self.LI_emb(hist_item_ids)
        li_emb = hist_item_embed.sum(dim=1, keepdim=True)
        all_il_emb = self.IL_emb.weight
        fmc = torch.matmul(li_emb,all_il_emb.transpose(0,1))

        predict = mf+fmc
        target_embeds = self.IL_emb(target_ids)

        scores = torch.mul(predict, target_embeds).sum(dim=2, keepdim=False)#计算分数
        return scores
'''



