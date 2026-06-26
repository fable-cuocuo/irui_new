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
            return rec_loss, modified_index, log_feats
        if self.config['rec_model'] == 'IOSC':
            neg_targets = neg_targets.type(torch.long)
            pos_target = pos_target.unsqueeze(1)
            neg_num = neg_targets.size()[1]
            common_prefer, modified_index, personal_prefer = self.context_encoder(user_id, hist_item_ids, num_weight_forget)
            common_score = self.calculate_score(common_prefer, user_id, hist_item_ids, pos_target, flag='pos')
            neg_score_1 = self.calculate_score(common_prefer, user_id, hist_item_ids, neg_targets, 'neg')
            rec_loss_ISC = -(torch.log(torch.sigmoid(common_score - neg_score_1)) / neg_num).sum(dim=1, keepdim=False)
            personal_score = self.calculate_score(personal_prefer, user_id, hist_item_ids, pos_target, flag='pos')
            neg_score_2 = self.calculate_score(personal_prefer, user_id, hist_item_ids, neg_targets, 'neg')
            rec_loss_OSC = -(torch.log(torch.sigmoid(personal_score  - neg_score_2)) / neg_num).sum(dim=1, keepdim=False)

            loss_compare = rec_loss_ISC - rec_loss_OSC
            loss_sum = rec_loss_ISC + rec_loss_OSC
            return loss_sum, rec_loss_ISC, rec_loss_OSC, modified_index, common_prefer, personal_prefer

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

        if self.config['rec_model'] == 'SASRec' or self.config['rec_model'] == 'IOSC':
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

        if self.config['rec_model'] == 'IOSC':
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

    def seqs_correction(self, input_seqs, current_pre, targets, changed_index, candidate_items):
        # hist_item_ids, current_pre, pos_target, changed_choice, train_candidates
        # input_seqs: bs,input_len
        # item_emb : item_num, hidden
        # changed_index: -2 表示修改target， -1 表示不修改 ，
        if torch.cuda.is_available():
            input_seqs = input_seqs.to(torch.device('cuda'))
            targets = targets.to(torch.device('cuda'))
            current_pre = current_pre.to(torch.device('cuda'))
            changed_index = changed_index.to(torch.device('cuda'))
            candidate_items = candidate_items.to(torch.device('cuda'))

        modified_seqs = input_seqs.clone()
        modified_targets = targets.clone()
        user_embed, item_embed = self.obtain_embeds(is_training=False)
        candidate_emb = item_embed[candidate_items]
        changed_target = torch.where(changed_index == -2.0)[0]
        hist_changed = torch.where(changed_index > 0)[0]
        modified_target_num = 0
        modified_hist_num = 0

        if len(changed_target) > 0:
        # 修改targets
            seqs_index = changed_target
            #[len_C_T, candidate_num, hidden_size]
            candidate_lists = candidate_emb[seqs_index]
            #[len_C_T, hidden_size]
            refer_item_emb = current_pre[seqs_index]
            output = torch.matmul(refer_item_emb.unsqueeze(1), candidate_lists.transpose(1, 2)).squeeze(dim=-1)
            t_c_index = output.argmax(-1)
            candidate_tmp = candidate_items[seqs_index]
            # modified_targets[seqs_index] = torch.index_select(candidate_tmp, 0, t_c_index)
            modified_targets[seqs_index] = torch.gather(candidate_tmp, 1, t_c_index).squeeze(dim=1)
            modified_target_num = len(changed_target)

        if len(hist_changed) > 0:
        # 修改modified_seqs
            instance_index = hist_changed
            modified_item_index = changed_index[instance_index].long()
            org_target = targets[instance_index]
            org_target_emb = item_embed[org_target]
            candidate_lists = candidate_emb[instance_index]
            output = torch.matmul(org_target_emb.unsqueeze(1), candidate_lists.transpose(1, 2)).squeeze(dim=-1)
            item_c_index = output.argmax(-1)#[num,1]
            cadidate_tmp_2 = candidate_items[instance_index]#[num, 3]
            modified_seqs_hist = modified_seqs[instance_index] #[num, 5]
            item_tmp = torch.gather(cadidate_tmp_2, 1, item_c_index)#[num, 1]
            modified_seqs_hist[modified_item_index.unsqueeze(0)] = item_tmp
            modified_seqs[instance_index] = modified_seqs_hist
            modified_hist_num = len(hist_changed)

        return modified_seqs, modified_targets, modified_target_num, modified_hist_num


'''
def seqs_correction(self, input_seqs, current_pre, targets, neg_targets, changed_index, candidate_items):
        # input_seqs: bs,input_len
        # item_emb : item_num, hidden
        # changed_index: -2 表示修改target， -1 表示不修改 ，
        if torch.cuda.is_available():
            input_seqs = input_seqs.to(torch.device('cuda'))
            targets = targets.to(torch.device('cuda'))
            all_neg_targets = neg_targets.to(torch.device('cuda'))
            current_pre = current_pre .to(torch.device('cuda'))
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

        i = 0
        while i < len(changed_target):
            seqs_index = changed_target[i]
            candidate_lists = candidate_emb[seqs_index]
            refer_item_emb = current_pre[seqs_index]
            # 修改target的参照item，从candidate里选择和这个item最像的
            # refer_item_emb = item_embed[refer_item]
            output = torch.matmul(refer_item_emb, candidate_lists.T)
            t_c_index = output.argmax(-1, keepdim=True)
            if candidate_lists[t_c_index] == 0:
                t_c_index = output.argmax(-2, keepdim=True)
            if candidate_items[seqs_index][t_c_index] != 0:
                modified_targets[seqs_index] = candidate_items[seqs_index][t_c_index]
                modified_target_num += 1
            i += 1

        j = 0
        while j < len(hist_changed):
            # 实例所在的index
            instance_index = hist_changed[j] # 0
            neg_targets = all_neg_targets[instance_index]
            # 要修改的item在instance里的index
            modified_item_index = changed_index[instance_index].type(torch.long)
            # 以target为参照item
            org_target = targets[instance_index]
            org_target_emb = item_embed[org_target]
            candidate_lists = candidate_emb[instance_index]
            output = torch.matmul(org_target_emb, candidate_lists.T)
            item_c_index = output.argmax(-1, keepdim=True)
            if candidate_lists[item_c_index] == 0:
                item_c_index = output.argmax(-2, keepdim=True)
            # modified_seqs[bs,input_len]  candidate_items [bs,candidate_nums]
            # org seqs
            if candidate_items[instance_index][item_c_index] != 0:
                modified_seqs[instance_index][modified_item_index] = candidate_items[instance_index][item_c_index]
                modified_hist_num += 1
                # print(target[seqs_index])
            j += 1

        return modified_seqs, modified_targets, modified_target_num, modified_hist_num


'''


def _init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
