import torch

from model.IOSCencoder import IOSC_encoder


class IOSC_encoder_Fuzzy(IOSC_encoder):
    """
    IOSC encoder with fuzzy-clustering-based natural multi-interest modeling.

    This class keeps the original IOSC forward behavior compatible, and adds:
    1) FCM memberships over user interacted items
    2) Interest-cluster overlap estimation
    3) Adaptive orthogonal regularization whose intensity is controlled by overlap
    """

    def __init__(self, config):
        super().__init__(config)
        self.fuzzy_num_clusters = int(config.get("fuzzy_num_clusters", 4))
        self.fuzzy_m = float(config.get("fuzzy_m", 2.0))
        self.fuzzy_eps = float(config.get("fuzzy_eps", 1e-8))
        self.orth_base = float(config.get("orth_base", 1.0))
        self.orth_lambda = float(config.get("orth_lambda", 0.05))
        self.interest_temperature = float(config.get("interest_temperature", 1.0))
        self.reliability_threshold = float(config.get("reliability_threshold", 0.0))
        self.use_interest_specific_scorer = bool(config.get("use_interest_specific_scorer", True))
        self.salience_temperature = float(config.get("salience_temperature", 1.0))

        hidden_size = int(config["hidden_size"])
        # ComiRec-style learnable interest queries.
        self.interest_queries = torch.nn.Parameter(torch.randn(self.fuzzy_num_clusters, hidden_size) * 0.02)
        self.interest_scorer_scale = torch.nn.Parameter(torch.ones(self.fuzzy_num_clusters, hidden_size))
        # Item representative score table (e.g., PageRank/centrality). Default: 1.0.
        self.item_importance = torch.nn.Embedding(config["item_num"], 1, padding_idx=0)
        torch.nn.init.ones_(self.item_importance.weight)
        self.item_importance.weight.data[0].fill_(0.0)
        if "item_importance_scores" in config:
            scores = torch.tensor(config["item_importance_scores"], dtype=self.item_importance.weight.dtype)
            max_len = min(scores.numel(), self.item_importance.weight.size(0))
            self.item_importance.weight.data[:max_len, 0] = scores[:max_len]
            self.item_importance.weight.data[0].fill_(0.0)

        # Cached tensors from latest forward.
        self.latest_membership = None
        self.latest_interest_vectors = None
        self.latest_cluster_overlap = None
        self.latest_adaptive_orth_loss = None
        self.latest_alpha = None
        self.latest_beta = None
        self.latest_interest_salience = None

    def _masked_softmax(self, logits, mask, dim):
        """
        logits: arbitrary shape
        mask: bool mask with same broadcastable shape as logits
        """
        masked_logits = logits.masked_fill(~mask, -1e4)
        probs = torch.softmax(masked_logits, dim=dim)
        probs = probs * mask.to(probs.dtype)
        probs = probs / (probs.sum(dim=dim, keepdim=True) + self.fuzzy_eps)
        return probs

    def _compute_interest_vectors_from_membership(self, item_embs, membership, valid_mask):
        """
        Weighted pooling with fuzzy membership (FCM-style m power).
        item_embs: [B, L, D], membership: [B, L, K], valid_mask: [B, L]
        return centers: [B, K, D]
        """
        um = membership.pow(self.fuzzy_m) * valid_mask.unsqueeze(-1)
        numerator = (um.unsqueeze(-1) * item_embs.unsqueeze(2)).sum(dim=1)  # [B, K, D]
        denominator = um.sum(dim=1).unsqueeze(-1) + self.fuzzy_eps  # [B, K, 1]
        return numerator / denominator

    def attention_membership_cluster(self, seq_feats, hist_item_ids):
        """
        Build multi-interest from attention logits:
        - beta_{t,k}: softmax over interests for each item t (item->interest assignment)
        - alpha_{k,t}: softmax over items for each interest k (interest->item aggregation)

        seq_feats: [B, L, D]
        returns:
            beta: [B, L, K]
            alpha: [B, K, L]
            centers: [B, K, D]
        """
        item_embs = self.item_emb(hist_item_ids)  # for membership-weighted centers
        valid_mask = hist_item_ids.ne(0)
        batch_size, _, hidden_size = item_embs.size()

        queries = self.interest_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]
        scale = float(hidden_size) ** 0.5
        logits = torch.matmul(seq_feats, queries.transpose(1, 2)) / (scale * self.interest_temperature)  # [B,L,K]

        beta_mask = valid_mask.unsqueeze(-1).expand_as(logits)  # [B,L,K]
        beta = self._masked_softmax(logits, beta_mask, dim=-1)

        alpha_logits = logits.transpose(1, 2)  # [B,K,L]
        alpha_mask = valid_mask.unsqueeze(1).expand_as(alpha_logits)  # [B,K,L]
        alpha = self._masked_softmax(alpha_logits, alpha_mask, dim=-1)

        centers = self._compute_interest_vectors_from_membership(item_embs, beta, valid_mask)
        return beta, alpha, centers

    def _compute_modified_index_from_multi_interest(self, beta, alpha, valid_mask):
        """
        Reliability per item t:
            r_t = max_k beta_{t,k} * max_k alpha_{k,t}
        Low reliability indicates uncertain assignment and low contribution.
        """
        item_assign_conf = beta.max(dim=-1)[0]           # [B,L]
        item_contrib_conf = alpha.max(dim=1)[0]          # [B,L]
        reliability = item_assign_conf * item_contrib_conf

        masked_reliability = torch.where(
            valid_mask,
            reliability,
            torch.full_like(reliability, float("inf"))
        )
        min_reliability, min_index = masked_reliability.min(dim=-1)
        no_valid = ~valid_mask.any(dim=-1)
        min_index[no_valid] = -1
        # If all interactions are reliable enough, do not modify.
        min_index[min_reliability > self.reliability_threshold] = -1
        return min_index.unsqueeze(1)

    def _compute_interest_salience(self, alpha, hist_item_ids):
        """
        Interest salience from representative items:
            sal_k = softmax_k( sum_t alpha_{k,t} * importance(i_t) )
        """
        importance = self.item_importance(hist_item_ids).squeeze(-1)  # [B, L]
        salience_logits = torch.sum(alpha * importance.unsqueeze(1), dim=-1)  # [B, K]
        salience = torch.softmax(salience_logits / self.salience_temperature, dim=-1)
        return salience

    def compute_cluster_overlap(self, membership):
        """
        Compute pairwise overlap between fuzzy clusters.
        membership: [B, L, K]
        return overlap: [B, K, K], in [0, 1]
        """
        # [B, K, L]
        u = membership.transpose(1, 2)
        min_u = torch.minimum(u.unsqueeze(2), u.unsqueeze(1)).sum(dim=-1)
        max_u = torch.maximum(u.unsqueeze(2), u.unsqueeze(1)).sum(dim=-1)
        overlap = min_u / (max_u + self.fuzzy_eps)
        return overlap

    def adaptive_orthogonal_loss(self, interest_vectors, overlap):
        """
        Adaptive orth loss:
        - Keep unrelated interests orthogonal (high penalty when overlap low)
        - Allow related interests soft coupling (low penalty when overlap high)

        interest_vectors: [B, K, D]
        overlap: [B, K, K]
        """
        k = interest_vectors.size(1)
        if k <= 1:
            return interest_vectors.new_zeros(())

        vec = torch.nn.functional.normalize(interest_vectors, p=2, dim=-1, eps=self.fuzzy_eps)
        sim = torch.matmul(vec, vec.transpose(1, 2))  # [B, K, K]

        eye = torch.eye(k, device=interest_vectors.device, dtype=interest_vectors.dtype).unsqueeze(0)
        off_diag = 1.0 - eye

        # Overlap high -> smaller orth penalty, overlap low -> stronger orth penalty
        weight = self.orth_base * (1.0 - overlap) * off_diag
        weighted_sim2 = weight * sim.pow(2)

        denom = off_diag.sum() * max(interest_vectors.size(0), 1)
        return weighted_sim2.sum() / (denom + self.fuzzy_eps)

    def fuse_interest_vectors(self, user_embed, interest_vectors):
        """
        Fuse K interests to one vector for legacy IOSC compatibility.
        Routing query uses user embedding.
        """
        route_score = torch.sum(interest_vectors * user_embed.unsqueeze(1), dim=-1)  # [B, K]
        route_weight = torch.softmax(route_score, dim=-1)  # [B, K]
        fused = torch.sum(route_weight.unsqueeze(-1) * interest_vectors, dim=1)  # [B, D]
        return fused

    def forward(self, users, hist_item_ids, num_weight_forget, return_aux=False):
        # Personal branch keeps IOSC original behavior.
        personal_prefer = self.OSCencoder(users, hist_item_ids)

        # Common branch is now multi-interest first.
        seq_feats, _ = self.hist2feats(hist_item_ids, num_weight_forget)
        beta, alpha, centers = self.attention_membership_cluster(seq_feats, hist_item_ids)
        valid_mask = hist_item_ids.ne(0)
        modified_index = self._compute_modified_index_from_multi_interest(beta, alpha, valid_mask)
        overlap = self.compute_cluster_overlap(beta)
        orth_loss = self.adaptive_orthogonal_loss(centers, overlap)

        self.latest_membership = beta
        self.latest_interest_vectors = centers
        self.latest_cluster_overlap = overlap
        self.latest_adaptive_orth_loss = self.orth_lambda * orth_loss
        self.latest_alpha = alpha
        self.latest_beta = beta
        self.latest_interest_salience = self._compute_interest_salience(alpha, hist_item_ids)

        user_embed = self.user_emb(users)
        common_prefer = self.fuse_interest_vectors(user_embed, centers)

        if return_aux:
            return (
                common_prefer,
                modified_index,
                personal_prefer,
                self.latest_adaptive_orth_loss,
                centers,
                overlap,
                beta,
            )

        # Keep signature compatible with original IOSC encoder.
        return common_prefer, modified_index, personal_prefer

    def get_adaptive_orth_loss(self):
        """
        Retrieve adaptive orth loss from latest forward.
        """
        if self.latest_adaptive_orth_loss is None:
            return torch.tensor(0.0, device=self.item_emb.weight.device, dtype=self.item_emb.weight.dtype)
        return self.latest_adaptive_orth_loss

    def get_multi_interest_vectors(self):
        """
        Access the latest multi-interest vectors [B, K, D] for retrieval/reranking stage.
        """
        return self.latest_interest_vectors

    def score_items_with_interests(self, interest_vectors, item_embs):
        """
        Score candidate items with multi-interest vectors.
        interest_vectors: [B, K, D]
        item_embs: [B, N, D]
        returns:
            scores: [B, N] (max over interests)
            scores_k: [B, K, N]
        """
        if self.use_interest_specific_scorer:
            scorer_scale = self.interest_scorer_scale.unsqueeze(0)  # [1, K, D]
            interest_vectors = interest_vectors * scorer_scale

        scores_k = torch.einsum("bkd,bnd->bkn", interest_vectors, item_embs)
        # Salience-aware scoring: emphasize interests supported by representative items.
        if self.latest_interest_salience is not None:
            scores_k = scores_k * self.latest_interest_salience.unsqueeze(-1)
        scores = scores_k.max(dim=1)[0]
        return scores, scores_k
