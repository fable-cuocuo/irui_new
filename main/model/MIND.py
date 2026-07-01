import torch
from torch import nn

from model.SeqRec import ContextEncoder


class MIND_encoder(ContextEncoder):
    """
    MIND-style multi-interest encoder adapted to current SeqRec pipeline.
    """

    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = int(config["hidden_size"])
        self.num_interest = int(config.get("num_interest", 4))
        self.routing_iters = int(config.get("mind_routing_iters", 3))
        self.max_seq_len = int(config.get("input_len", config.get("max_seq_len", 10)))
        self.routing_eps = float(config.get("mind_routing_eps", 1e-9))
        self.add_pos = bool(config.get("mind_add_pos", True))

        self.pre_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.post_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )

        self.routing_logits = nn.Parameter(
            torch.randn(self.num_interest, self.max_seq_len)
        )
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_size)

    def _build_mask(self, hist_item_ids, masks=None):
        if masks is None:
            return (hist_item_ids != 0).float()
        return masks.float()

    def _squash(self, caps):
        norm = torch.norm(caps, dim=-1, keepdim=True).clamp_min(self.routing_eps)
        norm_sq = norm * norm
        scale = norm_sq / ((1.0 + norm_sq) * norm + self.routing_eps)
        return scale * caps

    def forward(self, hist_item_ids, masks=None, return_all=False):
        seq_mask = self._build_mask(hist_item_ids, masks)  # [B, L]
        batch_size, seq_len = hist_item_ids.size()

        hist_emb = self.item_emb(hist_item_ids)  # [B, L, D]
        if self.add_pos:
            positions = torch.arange(
                seq_len, device=hist_item_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
            hist_emb = hist_emb + self.pos_emb(positions)

        hist_proj = self.pre_proj(hist_emb)  # [B, L, D]

        base_logits = self.routing_logits[:, :seq_len].unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, K, L]
        mask = seq_mask.unsqueeze(1) > 0
        drop_value = torch.full_like(base_logits, -1e9)
        routing_logits = base_logits

        interest_emb = None
        for step in range(self.routing_iters):
            masked_logits = torch.where(mask, routing_logits, drop_value)
            routing_weights = torch.softmax(masked_logits, dim=-1)
            routing_weights = routing_weights * seq_mask.unsqueeze(1)
            weight_sum = routing_weights.sum(dim=-1, keepdim=True).clamp_min(
                self.routing_eps
            )
            routing_weights = routing_weights / weight_sum

            interest_emb = torch.matmul(routing_weights, hist_proj)  # [B, K, D]
            interest_emb = self._squash(interest_emb)

            if step < self.routing_iters - 1:
                routing_logits = routing_logits + torch.matmul(
                    interest_emb, hist_proj.transpose(1, 2)
                )

        interest_emb = self.post_mlp(interest_emb)
        if return_all:
            return interest_emb
        return interest_emb[:, 0, :]
