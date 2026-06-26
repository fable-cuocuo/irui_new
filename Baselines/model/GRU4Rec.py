from model.SeqRec import ContextEncoder
from torch import nn


class GRU4Rec_encoder(ContextEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.network_param_init(config)
        print('###################GRU4Rec ENCODER YES###############')

    def network_param_init(self, config):
        self.seq_module = nn.GRU(input_size=config['hidden_size'], hidden_size=config['hidden_size'],
                          num_layers=config['gru_layer_num'], dropout=config['drop_ratio'],
                          batch_first=True)

    def seq_modelling(self, hist_item_ids, user_embed=None, rectify=True):
        hist_item_embed = self.item_emb.weight[hist_item_ids]
        output, _ = self.seq_module(hist_item_embed)  # (batch, seq_len, hidden_size)
        if rectify:
            return output[:, -1, :].squeeze(1), output[:, -2, :].squeeze(1)
        else:
            return output[:, -1, :].squeeze(1)

