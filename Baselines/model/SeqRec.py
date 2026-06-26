import torch
from torch import nn
from torch.nn.init import xavier_normal_
import numpy as np
from torch.nn import TransformerEncoderLayer
from model_utils.BERT_SeqRec import BertModel
from model_utils.BERT_SeqRec import BertConfig


class ContextEncoder(nn.Module):
    def __init__(self, config):
        self.config = config
        super().__init__()
        self.user_emb = nn.Embedding(config['user_num'], config['hidden_size'])
        self.item_emb = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0)
        self.network_param_init(config)
        self.seq_module = None

    def network_param_init(self, config):
        pass

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        pass

    def forward(self, user_id, hist_item_ids, rectify=True):
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        """
        context_embed = self.seq_modelling(hist_item_ids, rectify=False)
        if self.config['add_user']:
            user_embed = self.user_emb.weight[user_id]
            context_embed = context_embed + user_embed
        return context_embed

    def obtain_embeds(self):
        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight

        return users_emb, items_emb

class SeqRec(nn.Module):

    def __init__(self, config, context_encoder: ContextEncoder):
        super().__init__()
        self.config = config
        self.seq_module = None
        self.network_param_init(config)
        if self.config['train_type'] == 'train':
            # print('yes')
            self.context_encoder = context_encoder
            self.apply(_init_weights)

    def network_param_init(self, config):  # 设置seq_module的参数
        bert_config = BertConfig(config['item_num'], config)  # bert的词典大小为item_num 其余遵循config
        self.seq_module = BertModel(bert_config, use_outer_embed=True)  # outer_embed为true，其实是

    def set_encoder(self, encoder):
        print('yes')
        self.context_encoder = encoder

    def forward(self, user_id, hist_item_ids, masks, pos_target, neg_targets, sample_idices):

        # ??, sample_idices
        if torch.cuda.is_available():
            user_id = user_id.to(torch.device('cuda'))
            hist_item_ids = hist_item_ids.to(torch.device('cuda'))
            masks = masks.to(torch.device('cuda'))
            pos_target = pos_target.to(torch.device('cuda'))
            neg_targets = neg_targets.to(torch.device('cuda')).type(torch.long)
        user_embeds, item_embeds = self.context_encoder.obtain_embeds()
        neg_num = neg_targets.size()[1]
        # [bs, hidden_size]
        context_embed = self.context_encoder(user_id, hist_item_ids)
        # [bs, 1]
        pos_score = torch.mul(context_embed, item_embeds[pos_target]).sum(dim=1, keepdim=True)
        # [bs, neg_num, hidden_size]
        neg_targets = neg_targets.type(torch.long)
        neg_embeds = item_embeds[neg_targets]
        # [bs, neg_num]
        neg_scores = torch.mul(context_embed.unsqueeze(1), neg_embeds).sum(dim=2, keepdim=False)
       
        loss = -(torch.log(torch.sigmoid(pos_score - neg_scores[:, 0:2])))

        loss = loss.sum(dim=1, keepdim=True)

        return loss


    def recommender_forward(self, hist_item_ids, masks):
        if torch.cuda.is_available():
            hist_item_ids = hist_item_ids.to(torch.device('cuda'))
            masks = masks.double().to(torch.device('cuda'))

        user_embeds, item_embeds = self.obtain_embeds(is_training=True)
        bert_context = self.seq_module(hist_item_ids, attention_mask=masks, outer_embed=item_embeds)

        return bert_context[:, -1, :].squeeze(1)

    def recommender_loss(self, recommender_output, pos_target, neg_targets):

        if torch.cuda.is_available():
            pos_target = pos_target.to(torch.device('cuda'))
            neg_targets = neg_targets.to(torch.device('cuda'))

        user_embeds, item_embeds = self.obtain_embeds(is_training=True)
        pos = item_embeds[pos_target].squeeze()
        pos_score = torch.mul(recommender_output, pos).sum(dim=1, keepdim=True)

        neg_targets = neg_targets.type(torch.long)
        neg = item_embeds[neg_targets]
        neg_score = torch.mul(recommender_output.unsqueeze(1), neg).sum(dim=2, keepdim=False)

        neg_num = neg_targets.size()[1]

        rec_loss = -(torch.log(torch.sigmoid(pos_score - neg_score)) / neg_num).sum(dim=1, keepdim=False)

        return rec_loss

    def eval_ranking(self, test_batch):
        user_id, hist_item_ids, masks, target_ids = test_batch
        masks = masks.float()
        if torch.cuda.is_available():
            user_id = user_id.to(torch.device('cuda'))
            hist_item_ids = hist_item_ids.to(torch.device('cuda'))
            masks = masks.double().to(torch.device('cuda'))
            target_ids = target_ids.to(torch.device('cuda'))
            # print(f'eval_ranking: {user_id.is_cuda}')
        """
        user_id = [bs]
        hist_item_ids = [bs, seq_len]
        masks = [bs, seq_len]
        target_ids = [bs, eval_neg_num + 2]
        
        print(user_id.size())
        print(hist_item_ids.size())
        print(target_ids.size())
        print(user_id,hist_item_ids,target_ids)
        """

        # 第一次argsort知道了评分从高到低的索引值
        # 第二次argsort知道了索引为0，也就是真正的target，在评分里排第几。排越高，越准。
        output = self.context_encoder(user_id,hist_item_ids)
        user_embeds, item_embeds = self.context_encoder.obtain_embeds()
        target_ids = target_ids.type(torch.long)
        target_emb = item_embeds[target_ids]
        ranks = torch.mul(output.unsqueeze(1), target_emb).sum(dim=2, keepdim=False)
        ranks = ranks.argsort(dim=1, descending=True).argsort(dim=1, descending=False)[:, 0:1].float()

        '''ranks1 =  self.calculate_score(user_id, hist_item_ids, target_ids)
        print(ranks1.size())
        print(ranks1)
        ranks2 = ranks1.argsort(dim=1,descending=True)
        print(ranks2.size())
        print(ranks2)
        ranks3 = ranks2.argsort(dim=1, descending=False)
        print(ranks3.size())
        print(ranks3)
        ranks4 = ranks3[:, 0:1].float()
        print(ranks4.size())
        print(ranks4)

        ranks = ranks4
        '''

        # evaluate ranking
        metrics = {
            'ndcg_1': 0,
            'ndcg_5': 0,
            'ndcg_10': 0,
            'ndcg_20': 0,
            'hit_1': 0,
            'hit_5': 0,
            'hit_10': 0,
            'hit_20': 0,
            'ap': 0,
        }
        for rank in ranks:
            if rank < 1:
                metrics['ndcg_1'] += 1
                metrics['hit_1'] += 1
            if rank < 5:
                metrics['ndcg_5'] += 1 / torch.log2(rank + 2)
                metrics['hit_5'] += 1
            if rank < 10:
                metrics['ndcg_10'] += 1 / torch.log2(rank + 2)
                metrics['hit_10'] += 1
            if rank < 20:
                metrics['ndcg_20'] += 1 / torch.log2(rank + 2)
                metrics['hit_20'] += 1
            metrics['ap'] += 1.0 / (rank + 1)
        return metrics


def _init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
