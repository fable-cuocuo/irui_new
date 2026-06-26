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

    def forward(self, user_id, hist_item_ids):
        raise NotImplementedError


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

    def forward(self, user_id, hist_item_ids, masks, pos_target, neg_targets, sample_idices, num_weight_forget=0.5):

        # ??, sample_idices
        if torch.cuda.is_available():
            user_id = user_id.to(torch.device('cuda'))
            hist_item_ids = hist_item_ids.to(torch.device('cuda'))
            masks = masks.to(torch.device('cuda'))
            pos_target = pos_target.to(torch.device('cuda'))
            neg_targets = neg_targets.to(torch.device('cuda')).type(torch.long)

        if self.config['rec_model'] == 'BERT':
            recommend_output = self.context_encoder(hist_item_ids, masks)
            rec_loss = self.recommender_loss(recommender_output=recommend_output, pos_target=pos_target,
                                             neg_targets=neg_targets)
            return rec_loss

        if self.config['rec_model'] == 'SASRec':
            neg_targets = neg_targets.type(torch.long)
            # user_prop_embeds, item_prop_embeds = self.propagate_embeds(is_training=True)
            # neg_targets = torch.squeeze(neg_targets,dim=1)
            pos_target = pos_target.unsqueeze(1)
            neg_num = neg_targets.size()[1]
            # user_embeds, item_embeds = self.obtain_embeds(is_training=True)
            log_feats, modified_index = self.context_encoder(hist_item_ids, num_weight_forget)

            pos_score = self.calculate_score(log_feats, user_id, hist_item_ids, pos_target, 'pos')
            # pos_score = torch.unsqueeze(pos_score, dim=1)
            # print(pos_score.size())
            neg_score = self.calculate_score(log_feats, user_id, hist_item_ids, neg_targets, 'neg')
            # print(neg_score.size())
            rec_loss = -(torch.log(torch.sigmoid(pos_score - neg_score)) / neg_num).sum(dim=1, keepdim=False)
            # print(rec_W_loss.size())
            return rec_loss, modified_index

    def obtain_embeds(self, is_training):
        users_emb = self.context_encoder.user_emb.weight
        items_emb = self.context_encoder.item_emb.weight

        return users_emb, items_emb

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

    def calculate_score(self, log_feats, user_id, hist_item_ids, target, flag):

        global modified_index, score
        user_embeds, item_embeds = self.obtain_embeds(is_training=True)
        if self.config['rec_model'] == 'FPMC':
            # print(self.context_encoder)
            user_emb, iu_emb, il_emb, li_emb = self.context_encoder(user_id, hist_item_ids, target)

            mf = torch.matmul(user_emb, iu_emb.permute(0, 2, 1))  # [b,1,emb]*[b,emb,n]
            mf = torch.squeeze(mf, dim=1)  # [b,1,n]
            # print(mf.size())

            fmc = torch.matmul(li_emb, il_emb.permute(0, 2, 1))  # [b,1,emb]*[b,emb,n]
            fmc = torch.squeeze(fmc, dim=1)
            # print(fmc.size())
            score = mf + fmc
            score = torch.squeeze(score)

            return score

        if self.config['rec_model'] == 'SASRec':
            if flag == 'pos':
                # [bs,50] [bs,1]
                pos = item_embeds[target].squeeze()
                score = torch.mul(log_feats, pos).sum(dim=1, keepdim=True)
            elif flag == 'neg':
                target = target.type(torch.long)
                neg = item_embeds[target]
                score = torch.mul(log_feats.unsqueeze(1), neg).sum(dim=2, keepdim=False)
            return score

        # return score(FPMC)

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
        user_embeds, item_embeds = self.obtain_embeds(is_training=True)
        target_ids = target_ids.type(torch.long)
        target_emb = item_embeds[target_ids]

        if self.config['rec_model'] == 'BERT':
            output = self.context_encoder(hist_item_ids, masks)

        if self.config['rec_model'] == 'SASRec':
            output = self.recommender_forward(hist_item_ids, masks)

        scores = torch.mul(output.unsqueeze(1), target_emb).sum(dim=2, keepdim=False)
        pos_score = scores[:, 0: 1]
        neg_scores = scores[:, 1: -1]
        ranks = (neg_scores > pos_score).long().sum(dim=1, keepdim=False)
        # ranks = ranks.argsort(dim=1, descending=True).argsort(dim=1, descending=False)[:, 0:1].float()

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

    def seqs_correction(self, input_seqs, masks, targets, neg_targets, changed_index, candidate_items):
        if torch.cuda.is_available():
            input_seqs = input_seqs.to(torch.device('cuda'))
            targets = targets.to(torch.device('cuda'))
            all_neg_targets = neg_targets.to(torch.device('cuda'))
            masks = masks.to(torch.device('cuda'))
            changed_index = changed_index.to(torch.device('cuda'))
            candidate_items = candidate_items.to(torch.device('cuda'))

        modified_seqs = input_seqs.clone()
        modified_targets = targets.clone()
        user_embed, item_embed = self.obtain_embeds(is_training=False)
        candidate_emb = item_embed[candidate_items]
        changed_target = torch.where(changed_index == -2.0)[0]
        no_changed = torch.where(changed_index == -1.0)[0]
        hist_changed = torch.where(changed_index > 0)[0]
        modified_target_num = 0
        modified_hist_num = 0
      
        if len(no_changed) == len(changed_index ):
            return modified_seqs, modified_targets, modified_target_num, modified_hist_num

        # Process changed_target
        seqs_indices = changed_target.unsqueeze(1)
        neg_targets = all_neg_targets[seqs_indices]
        candidate_lists = candidate_emb[seqs_indices]
        one_input = input_seqs[seqs_indices]
        one_mask = masks[seqs_indices]
        refer_item_emb = self.context_encoder.encoder(one_input)
        output = torch.matmul(refer_item_emb, candidate_lists.transpose(1, 2))
        t_c_index = output.argmax(-1, keepdim=True)

        rec_output = self.recommender_forward(one_input, one_mask)
        changedTarget = candidate_items[seqs_indices, t_c_index]
        changedTarget = torch.cat([changedTarget, neg_targets], dim=1)
        target_emb = item_embed[changedTarget]

        scores = torch.mul(rec_output.unsqueeze(1), target_emb).sum(dim=2)
        scores[changedTarget == 0] = -100000
        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:-1]
        new_rank = (neg_scores > pos_score).long().sum(dim=1)

        org_target = targets[seqs_indices].unsqueeze(1)
        org_target = torch.cat([org_target, neg_targets], dim=1)
        org_target_emb = item_embed[org_target]
        org_scores = torch.mul(rec_output.unsqueeze(1), org_target_emb).sum(dim=2)
        org_scores[org_target == 0] = -100000
        org_pos_score = org_scores[:, 0:1]
        org_neg_scores = org_scores[:, 1:-1]
        org_rank = (org_neg_scores > org_pos_score).long().sum(dim=1)

        rise = new_rank - org_rank
        mask = (rise <= 0) & (candidate_items[seqs_indices, t_c_index] != 0)
        modified_targets[seqs_indices[mask]] = candidate_items[seqs_indices, t_c_index][mask]
        modified_target_num += mask.sum()

        # Process hist_changed
        instance_indices = hist_changed.unsqueeze(1)
        neg_targets = all_neg_targets[instance_indices]
        modified_item_index = changed_index[instance_indices].long()
        org_target = targets[instance_indices]
        org_target_emb = item_embed[org_target]
        candidate_lists = candidate_emb[instance_indices]
        output = torch.matmul(org_target_emb, candidate_lists.transpose(1, 2))
        item_c_index = output.argmax(-1, keepdim=True)

        org_seqs = modified_seqs[instance_indices]
        one_mask = masks[instance_indices].unsqueeze(0)
        org_rec_output = self.recommender_forward(org_seqs.unsqueeze(0), one_mask)

        new_seqs = modified_seqs[instance_indices].clone()
        new_seqs.scatter_(1, modified_item_index.unsqueeze(2), candidate_items[instance_indices, item_c_index].unsqueeze(1))
        new_rec_output = self.recommender_forward(new_seqs, one_mask)

        all_targets = torch.cat([org_target.unsqueeze(0).unsqueeze(0), neg_targets.unsqueeze(0)], dim=1)
        all_targets_embs = item_embed[all_targets]
        new_scores = torch.mul(new_rec_output.unsqueeze(1), all_targets_embs).sum(dim=2)
        new_scores[all_targets == 0] = -100000
        new_pos_score = new_scores[:, 0:1]
        new_neg_scores = new_scores[:, 1:-1]
        new_rank = (new_neg_scores > new_pos_score).long().sum(dim=1)

        scores = torch.mul(org_rec_output.unsqueeze(1), all_targets_embs).sum(dim=2)
        scores[all_targets == 0] = -100000
        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:-1]
        org_rank = (neg_scores > pos_score).long().sum(dim=1)

        rise = new_rank - org_rank
        mask = (rise <= 0) & (candidate_items[instance_indices, item_c_index] != 0)
        modified_seqs[instance_indices[mask], modified_item_index[mask]] = candidate_items[instance_indices, item_c_index][mask]
        modified_hist_num += mask.sum()

        return modified_seqs, modified_targets, modified_target_num, modified_hist_num




def _init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
