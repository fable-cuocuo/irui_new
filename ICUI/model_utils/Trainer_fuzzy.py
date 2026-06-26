import torch

from model_utils.Trainer import Trainer


class TrainerFuzzyOrth(Trainer):
    """
    Trainer variant for IOSC fuzzy encoder.
    Adds adaptive orthogonal regularization into total training loss
    without modifying the original Trainer implementation.
    """

    def _score_with_interests(self, multi_interests, item_embs):
        encoder = self._model_.context_encoder
        if hasattr(encoder, 'score_items_with_interests'):
            return encoder.score_items_with_interests(multi_interests, item_embs)
        scores_k = torch.einsum('bkd,bnd->bkn', multi_interests, item_embs)
        scores = scores_k.max(dim=1)[0]
        return scores, scores_k

    def _orth_warmup_scale(self, epoch_num, dtype):
        warmup_epochs = int(self.config.get('orth_warmup_epochs', 0))
        max_scale = float(self.config.get('orth_warmup_max_scale', 1.0))
        if warmup_epochs <= 0:
            return torch.tensor(max_scale, device=self._device, dtype=dtype)
        if epoch_num >= warmup_epochs:
            return torch.tensor(max_scale, device=self._device, dtype=dtype)
        # linearly ramp from 0 to max_scale in warmup epochs
        scale = max_scale * float(epoch_num + 1) / float(warmup_epochs)
        return torch.tensor(scale, device=self._device, dtype=dtype)

    def _target_aware_multi_interest_loss(self, multi_interests, pos_target, neg_targets):
        """
        Target-aware multi-interest scoring:
        score(u, i) = max_k <v_{u,k}, e_i>
        """
        _, item_embeds = self._model_.obtain_embeds(is_training=True)
        pos_target = pos_target.type(torch.long)
        neg_targets = neg_targets.type(torch.long)

        # [B, D]
        pos_emb = item_embeds[pos_target]
        # [B, K]
        _, pos_scores_k_3d = self._score_with_interests(multi_interests, pos_emb.unsqueeze(1))
        pos_scores_k = pos_scores_k_3d.squeeze(-1)  # [B, K]
        pos_scores = pos_scores_k.max(dim=1, keepdim=True)[0]  # [B, 1]

        # [B, N, D]
        neg_emb = item_embeds[neg_targets]
        neg_scores, _ = self._score_with_interests(multi_interests, neg_emb)  # [B, N]

        neg_num = neg_targets.size(1)
        rec_loss = -(torch.log(torch.sigmoid(pos_scores - neg_scores)) / neg_num).sum(dim=1, keepdim=False)
        return rec_loss

    def _target_aware_fused_prefer(self, multi_interests, pos_target):
        """
        Build a target-aware fused preference for correction stage.
        """
        _, item_embeds = self._model_.obtain_embeds(is_training=True)
        pos_target = pos_target.type(torch.long)
        pos_emb = item_embeds[pos_target]  # [B, D]
        route_score = torch.sum(multi_interests * pos_emb.unsqueeze(1), dim=-1)  # [B, K]
        route_weight = torch.softmax(route_score, dim=-1)  # [B, K]
        return torch.sum(route_weight.unsqueeze(-1) * multi_interests, dim=1)  # [B, D]

    def train_one_batch(self, user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_idices,
                        epoch_num):
        if self.config.get('rec_model') != 'IOSC':
            return super().train_one_batch(
                user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_idices, epoch_num
            )

        modified_target_num = 0
        modified_hist_num = 0

        self._model_.train()
        self._optimizer_.zero_grad()
        num_weight_forget = self.weight_forget_rates[epoch_num]

        loss_sum, rec_loss_isc, rec_loss_osc, modified_index, common_prefer, personal_prefer = self._model_(
            user_id,
            hist_item_ids,
            masks,
            pos_target,
            neg_targets,
            sample_idices,
            num_weight_forget
        )
        # Remove target-item unreliable identification/correction.
        # Only keep sequence-item correction based on modified_index from multi-interest reliability.
        modified_index = modified_index.squeeze()
        changed_choice = modified_index

        self._model_.eval()
        with torch.no_grad():
            correction_prefer = common_prefer
            encoder = self._model_.context_encoder
            if hasattr(encoder, 'get_multi_interest_vectors'):
                multi_interests = encoder.get_multi_interest_vectors()
                if multi_interests is not None:
                    correction_prefer = self._target_aware_fused_prefer(multi_interests, pos_target)
            modified_seqs, modified_target, modified_target_num, modified_hist_num = self._model_.seqs_correction(
                hist_item_ids, correction_prefer, pos_target, changed_choice, train_candidates
            )
        self._model_.train()

        modified_recommender_output = self._model_.recommender_forward(modified_seqs, masks)
        modified_recommender_loss = self._model_.recommender_loss(
            modified_recommender_output, modified_target, neg_targets
        )

        orth_loss = torch.tensor(0.0, device=self._device, dtype=loss_sum.dtype)
        multi_interest_loss = torch.tensor(0.0, device=self._device, dtype=loss_sum.dtype)
        encoder = self._model_.context_encoder
        if hasattr(encoder, 'get_multi_interest_vectors'):
            multi_interests = encoder.get_multi_interest_vectors()
            if multi_interests is not None:
                rec_loss_vec = self._target_aware_multi_interest_loss(multi_interests, pos_target, neg_targets)
                multi_interest_loss = rec_loss_vec.sum() * float(
                    self.config.get('multi_interest_loss_weight', 0.3)
                )
        if hasattr(encoder, 'get_adaptive_orth_loss'):
            orth_loss = encoder.get_adaptive_orth_loss()
            orth_loss = orth_loss * self._orth_warmup_scale(epoch_num, loss_sum.dtype)

        corr_loss = modified_recommender_loss.sum() * float(self.config.get('corr_loss_weight', 1.0))
        all_loss = (
            corr_loss
            + multi_interest_loss
            + orth_loss
        )
        all_loss.backward()
        self._optimizer_.step()

        return all_loss, rec_loss_isc, rec_loss_osc, modified_target_num, modified_hist_num
