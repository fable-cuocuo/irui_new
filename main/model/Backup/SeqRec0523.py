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

    def seqs_correction(self, input_seqs, masks, targets, changed_index, candidate_items):
        # input_seqs: bs,input_len
        # item_emb : item_num, hidden
        # changed_index: -2 表示修改target， -1 表示不修改 ，
        if torch.cuda.is_available():
            input_seqs = input_seqs.to(torch.device('cuda'))
            targets = targets.to(torch.device('cuda'))
            changed_index = changed_index.to(torch.device('cuda'))
            candidate_items = candidate_items.to(torch.device('cuda'))

        modified_seqs = input_seqs.clone()
        modified_targets = targets.clone()
        user_embed, item_embed = self.obtain_embeds(is_training=False)
        # [bs, 4, 50]
        candidate_emb = item_embed[candidate_items]
        changed_target = torch.where(changed_index == -2)[0]
        no_changed = torch.where(changed_index == -1)[0]
        hist_changed = torch.where(changed_index > 0)[0]
        modified_target_num = 0
        modified_hist_num = 0

        i = 0
        while i < len(changed_target):
            seqs_index = changed_target[i]
            org_target = targets[seqs_index]
            org_target_emb = item_embed[org_target]
            # [5,4,50]
            candidate_lists = candidate_emb[seqs_index]
            # hists = input_seqs[seqs_index]
            # changed_index = torch.randint(0, 5, [1])
            # refer_item = hists[changed_index]
            # 参照item是transformer的输出
            # 1,50
            one_input = input_seqs[seqs_index].unsqueeze(0)
            one_mask = masks[seqs_index].unsqueeze(0)
            if self.config['rec_model'] == 'COR_BERT':
                refer_item_emb, _ = self.context_encoder(one_input, masks)
                refer_item_emb = refer_item_emb[seqs_index]
            else:
                refer_item_emb = self.context_encoder.encoder(one_input)
            # 修改target的参照item，从candidate里选择和这个item最像的
            # refer_item_emb = item_embed[refer_item]
            output = torch.matmul(refer_item_emb, candidate_lists.T)
            t_c_index = output.argmax(-1, keepdim=True)
            # print(t_c_index)
            # print(candidate_items[seqs_index][t_c_index])

            chatarget = candidate_items[seqs_index][t_c_index]
            target_emb = item_embed[changed_target]
            rec_output = self.recommender_forward(one_input, one_mask)
            scores = torch.mul(rec_output.unsqueeze(1), target_emb).sum(dim=2, keepdim=False)
            pos_score = scores[:, 0: 1]
            neg_scores = scores[:, 1: -1]
            ranks = (neg_scores > pos_score).long().sum(dim=1, keepdim=False)
            if ranks < self.config['sampler_rank']:
                modified_targets[seqs_index] = chatarget
                modified_target_num += 1
                # print(target[seqs_index])
            i += 1

        j = 0
        while j < len(hist_changed):
            # 实例所在的index
            instance_index = hist_changed[j]  # 0
            # 要修改的item在instance里的index
            modified_item_index = changed_index[instance_index].type(torch.long)
            # 以target为参照item
            org_target = targets[instance_index]
            org_target_emb = item_embed[org_target]
            candidate_lists = candidate_emb[instance_index]
            output = torch.matmul(org_target_emb, candidate_lists.T)
            item_c_index = output.argmax(-1, keepdim=True)
            # print(t_c_index)
            # print(candidate_items[seqs_index][t_c_index])
            # modified_seqs[bs,input_len]  candidate_items [bs,candidate_nums]
            the_seqs = modified_seqs[instance_index]
            the_seqs[modified_item_index] = candidate_items[instance_index][item_c_index]
            the_seqs = the_seqs.unsqueeze(0)
            one_mask = masks[instance_index].unsqueeze(0)
            rec_output = self.recommender_forward(the_seqs, one_mask)
            scores = torch.mul(rec_output.unsqueeze(1), target_emb).sum(dim=2, keepdim=False)
            pos_score = scores[:, 0: 1]
            neg_scores = scores[:, 1: -1]
            ranks = (neg_scores > pos_score).long().sum(dim=1, keepdim=False)

            if ranks < self.config['sampler_rank']:
                modified_seqs[instance_index][modified_item_index] = candidate_items[instance_index][item_c_index]
                modified_hist_num += 1
                # print(target[seqs_index])
            j += 1

        return modified_seqs, modified_targets, modified_target_num, modified_hist_num


def _init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
