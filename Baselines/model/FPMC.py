from model.SeqRec import ContextEncoder
from torch import nn


class FPMC_encoder(ContextEncoder):

    def __init__(self, config):
        super().__init__(config)
        print('###################FPMC ENCODER YES###############')

    def network_param_init(self, config):
        self.seq_module = None

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        hist_item_embed = self.item_emb.weight[hist_item_ids]
        # [bs, hidden_size]
        sumed_item_embed = hist_item_embed.sum(dim=1, keepdim=False)

        if rectify:
            second_last_hist_item_embed = self.item_emb.weight[hist_item_ids[:, 1:-1]]
            sumed_item_embed_second_last = second_last_hist_item_embed.sum(dim=1, keepdim=False)
            return sumed_item_embed, sumed_item_embed_second_last
        else:
            return sumed_item_embed

