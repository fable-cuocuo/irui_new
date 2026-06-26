import torch
from torch import nn
from torch.nn.init import xavier_normal_
from model.SeqRec import ContextEncoder
import numpy as np


class SASContextEncoder(ContextEncoder):
    def __init__(self, config):
        # self.user_emb = nn.Embedding(config['user_num'], config['hidden_size'])
        # self.item_emb = nn.Embedding(config['item_num'], config['hidden_size'], padding_idx=0)
        super().__init__(config)

        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.5

        self.pos_emb = torch.nn.Embedding(config['input_len'], config['hidden_size'])
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # self.network_param_init(config)

        self.last_layernorm = torch.nn.LayerNorm(config['hidden_size'], eps=1e-8)

        for _ in range(self.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(config['hidden_size'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(config['hidden_size'],
                                                         self.num_heads,
                                                         self.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(config['hidden_size'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_size'], self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)



    def forward(self, hist_item_ids):
        raise NotImplementedError


class SASRec_encoder(SASContextEncoder):

    def __init__(self, config):
        super().__init__(config)
        print('------------SASREC ENCODER YES------------')

    def forward(self, hist_item_ids, num_weight_forget):
        # user_embed, item_embed = self.obtain_embeds(is_training=True)
        log_feats, modified_index = self.hist2feats(hist_item_ids, num_weight_forget)  # user_ids hasn't been used yet
        log_feats = log_feats[:, -1, :]
        # pos = item_embed[target].squeeze()
        # print(pos.shape)
        # score = torch.mul(log_feats, pos).sum(dim=1, keepdim=True)
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(torch.device('cuda')))

        # neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return log_feats, modified_index  # pos_pred, neg_pred

    def hist2feats(self, hist_item_ids, num_weight_forget):
        '''
        num_weight_forget: 低于均值的多少，被认为不可靠

        '''
        # hist_item_ids [bs,len]
        global mha_weights
        seqs = self.item_emb(hist_item_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        # positions = np.tile(np.array(range(hist_item_ids.shape[1])), [hist_item_ids.shape[0], 1])
        # seqs += self.pos_emb(torch.LongTensor(positions).to(torch.device('cuda')))
        positions = np.tile(np.array(range(hist_item_ids.shape[1])), (hist_item_ids.shape[0], 1))
        positions = torch.LongTensor(np.array(positions)).to(torch.device('cuda'))
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.eq(hist_item_ids, 0)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=torch.device('cuda')))

        # lowest weight index [0,0,0,0,0]
        weight = [0] * len(hist_item_ids)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            # Q:[input_len,bs,hidden_size] L,N,E
            Q = self.attention_layernorms[i](seqs)
            # mha_outputs : 5,256,50 (L,N,E) L:output sequence length，N:batchsize, E:hidden_size
            # mha_weights : 256,5,5  (N,L,S) S:source sequece length
            mha_outputs, mha_weights = self.attention_layers[i](Q, seqs, seqs,
                                                                attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        # 得到最终output相对于source的output，
        final_weight = mha_weights[:, -1, :]
        final_weight = final_weight.squeeze()
        mean = final_weight.mean(dim=1, keepdim=True)
        mask = final_weight < (mean * num_weight_forget)
        final_weight[final_weight == 0] += 0.001
        output = final_weight * mask
        min_index = torch.where(output == 0, torch.full_like(output, float('inf')), output).argmin(dim=-1)
        min_index[output.sum(dim=-1) == 0] = -1
        modified_index = min_index.unsqueeze(1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats, modified_index

    def encoder(self, hist_item_ids):
        # hist_item_ids [bs,len]
        seqs = self.item_emb(hist_item_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        # positions = np.tile(np.array(range(hist_item_ids.shape[1])), [hist_item_ids.shape[0], 1])
        # seqs += self.pos_emb(torch.LongTensor(positions).to(torch.device('cuda')))
        positions = np.tile(np.array(range(hist_item_ids.shape[1])), (hist_item_ids.shape[0], 1))
        positions = torch.LongTensor(np.array(positions)).to(torch.device('cuda'))
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.eq(hist_item_ids, 0)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=torch.device('cuda')))

        # lowest weight index [0,0,0,0,0]

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            # Q:[input_len,bs,hidden_size] L,N,E
            Q = self.attention_layernorms[i](seqs)
            # mha_outputs : 5,256,50 (L,N,E) L:output sequence length，N:batchsize, E:hidden_size
            # mha_weights : 256,5,5  (N,L,S) S:source sequece length
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)
        # 得到最终output相对于source的output，
        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats[:, -1, :]


def _init_weights(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)

    # ??


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
