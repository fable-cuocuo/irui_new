import torch
from torch import nn
from model.SeqRec import ContextEncoder


class ComiRec_encoder(ContextEncoder):
    """
    Lightweight ComiRec-SA style encoder in the current PyTorch pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config["hidden_size"]
        self.max_seq_len = config["input_len"]
        self.num_interest = int(config.get("num_interest", 4))
        self.add_pos = bool(config.get("comirec_add_pos", True))

        self.item_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 4, self.num_interest),
        )
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_size)

    def _build_mask(self, hist_item_ids, masks=None):
        if masks is None:
            seq_mask = (hist_item_ids != 0).float()
        else:
            seq_mask = masks.float()
        return seq_mask

    def forward(self, hist_item_ids, masks=None, return_all=False):
        seq_mask = self._build_mask(hist_item_ids, masks)  # [B, L]

        item_seq_emb = self.item_emb(hist_item_ids)  # [B, L, D]
        if self.add_pos:
            positions = torch.arange(
                hist_item_ids.size(1), device=hist_item_ids.device
            ).unsqueeze(0).expand(hist_item_ids.size(0), -1)
            item_seq_emb = item_seq_emb + self.pos_emb(positions)

        att_logits = self.item_proj(item_seq_emb).transpose(1, 2)  # [B, K, L]
        att_logits = att_logits.masked_fill(seq_mask.unsqueeze(1) == 0, -1e9)
        att = torch.softmax(att_logits, dim=-1)

        # Keep masked positions at zero and renormalize for numerical stability.
        att = att * seq_mask.unsqueeze(1)
        att_sum = att.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        att = att / att_sum

        interest_emb = torch.matmul(att, item_seq_emb)  # [B, K, D]

        if return_all:
            return interest_emb
        return interest_emb[:, 0, :]
