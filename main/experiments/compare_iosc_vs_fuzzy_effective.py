import copy
import os
import random
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(CURRENT_DIR)
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

from data_utils.SeqDataGenerator import SeqDataCollector
import model_utils.Trainer as trainer_module
from model_utils.Trainer import Trainer
from model_utils.Trainer_fuzzy import TrainerFuzzyOrth
from model.IOSCencoder import IOSC_encoder
from model.IOSCencoder_fuzzy import IOSC_encoder_Fuzzy
from model.Bert4Rec import BERT_encoder
from model.ComiRec import ComiRec_encoder
from model.MIND import MIND_encoder
from steam_main import config as steam_config


def seed_all(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config():
    cfg = copy.deepcopy(steam_config)
    cfg["dataset"] = "ml1m"
    cfg["input_len"] = 10
    cfg["max_seq_len"] = 10
    cfg["epoch_num"] = 3
    cfg["save_epochs"] = []
    cfg["train_type"] = "train"
    cfg["rec_model"] = "IOSC"
    cfg["train_batch_size"] = 512
    cfg["test_batch_size"] = 512
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["fuzzy_num_clusters"] = 4
    cfg["interest_temperature"] = 1.0
    cfg["use_interest_fusion"] = False
    cfg["use_interest_specific_scorer"] = True
    cfg["orth_lambda"] = 0.01
    cfg["orth_warmup_epochs"] = 1
    cfg["orth_warmup_max_scale"] = 1.0
    cfg["multi_interest_loss_weight"] = 0.3
    cfg["corr_loss_weight"] = 1.0
    cfg["reliability_threshold"] = 0.15
    return cfg


def patch_iosc_device_behavior():
    def patched_hist2feats(self, hist_item_ids, num_weight_forget):
        seqs = self.item_emb(hist_item_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(hist_item_ids.shape[1])), (hist_item_ids.shape[0], 1))
        device = hist_item_ids.device
        positions = torch.LongTensor(np.array(positions)).to(device)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.eq(hist_item_ids, 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            q = self.attention_layernorms[i](seqs)
            mha_outputs, mha_weights = self.attention_layers[i](q, seqs, seqs, attn_mask=attention_mask)
            seqs = q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        final_weight = mha_weights[:, -1, :]
        final_weight = final_weight.squeeze()
        mean = final_weight.mean(dim=1, keepdim=True)
        mask = final_weight < (mean * num_weight_forget)
        final_weight[final_weight == 0] += 0.001
        output = final_weight * mask
        min_index = torch.where(output == 0, torch.full_like(output, float("inf")), output).argmin(dim=-1)
        min_index[output.sum(dim=-1) == 0] = -1
        modified_index = min_index.unsqueeze(1)
        log_feats = self.last_layernorm(seqs)
        return log_feats, modified_index

    IOSC_encoder.hist2feats = patched_hist2feats
    IOSC_encoder_Fuzzy.hist2feats = patched_hist2feats


class QuickRunMixin:
    def __init__(self, *args, max_train_batches=80, **kwargs):
        self.max_train_batches = max_train_batches
        super().__init__(*args, **kwargs)

    def run_co(self):
        for epoch in range(self.config["epoch_num"]):
            self._model_.train()
            loss_iter = 0.0
            rec_loss_isc_iter = 0.0
            rec_loss_osc_iter = 0.0
            seen_batch = 0
            for i, batch in enumerate(self.train_loader):
                if i >= self.max_train_batches:
                    break
                user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_indices = batch
                if self.config["rec_model"] == "IOSC":
                    loss, rec_loss_isc, rec_loss_osc, _, _ = self.train_one_batch(
                        user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_indices, epoch
                    )
                else:
                    loss, _, _ = self.train_one_batch(
                        user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_indices, epoch
                    )
                    rec_loss_isc = torch.tensor(0.0)
                    rec_loss_osc = torch.tensor(0.0)
                loss_iter += float(loss.item())
                rec_loss_isc_iter += float(rec_loss_isc.sum().item())
                rec_loss_osc_iter += float(rec_loss_osc.sum().item())
                seen_batch += 1
            if seen_batch > 0:
                print(
                    f"[epoch {epoch}] avg_loss={loss_iter/seen_batch:.4f}, "
                    f"avg_isc={rec_loss_isc_iter/seen_batch:.4f}, avg_osc={rec_loss_osc_iter/seen_batch:.4f}, "
                    f"seen_batch={seen_batch}"
                )
            self.evaluate(epoch)


class QuickTrainerBase(QuickRunMixin, Trainer):
    pass


class QuickTrainerFuzzy(QuickRunMixin, TrainerFuzzyOrth):
    pass


def patch_short_epoch_forget_schedule():
    trainer_module.Trainer.build_high_forget_rates = (
        lambda self: np.ones(self.config["epoch_num"]) * self.config["high_loss_drop"]
    )
    trainer_module.Trainer.build_low_forget_rates = (
        lambda self: np.ones(self.config["epoch_num"]) * self.config["low_loss_drop"]
    )
    trainer_module.Trainer.build_weight_forget_rates = (
        lambda self: np.ones(self.config["epoch_num"]) * self.config["weight_drop"]
    )


def build_and_train(
    trainer_cls,
    encoder_cls,
    orth_lambda=None,
    warmup_epochs=None,
    rec_model="IOSC",
    corr_loss_weight=None,
):
    cfg = build_config()
    cfg["rec_model"] = rec_model
    if orth_lambda is not None:
        cfg["orth_lambda"] = float(orth_lambda)
    if warmup_epochs is not None:
        cfg["orth_warmup_epochs"] = int(warmup_epochs)
    if corr_loss_weight is not None:
        cfg["corr_loss_weight"] = float(corr_loss_weight)
    seed_all(123)
    patch_short_epoch_forget_schedule()
    if rec_model == "IOSC":
        patch_iosc_device_behavior()
    trainer_module.IOSC_encoder = encoder_cls
    data_model = SeqDataCollector(cfg)
    trainer = trainer_cls(cfg, data_model, save_dir="./datasets/" + cfg["dataset"] + "/seq/")
    trainer.run_co()
    return trainer


@torch.no_grad()
def evaluate_with_candidate_routing(model, test_batches, num_weight_forget=0.5, use_multi_interest=True):
    model.eval()
    current_metrics = {
        "ndcg_1": 0,
        "ndcg_5": 0,
        "ndcg_10": 0,
        "ndcg_20": 0,
        "hit_1": 0,
        "hit_5": 0,
        "hit_10": 0,
        "hit_20": 0,
        "ap": 0,
    }
    test_size = 0

    _, item_embeds = model.obtain_embeds(is_training=True)

    for test_batch in test_batches:
        user_id, hist_item_ids, _, target_ids = test_batch
        if torch.cuda.is_available():
            user_id = user_id.to(torch.device("cuda"))
            hist_item_ids = hist_item_ids.to(torch.device("cuda"))
            target_ids = target_ids.to(torch.device("cuda"))
        target_ids = target_ids.type(torch.long)

        # Trigger context encoder forward and build interest set.
        if model.config.get("rec_model") == "BERT":
            masks = (hist_item_ids != 0).double()
            single_interest = model.context_encoder(hist_item_ids, masks)  # [B, D]
            multi_interests = single_interest.unsqueeze(1)
        elif model.config.get("rec_model") in ("COMIREC", "MIND"):
            masks = (hist_item_ids != 0).double()
            multi_interests = model.context_encoder(hist_item_ids, masks, return_all=True)
        else:
            common_prefer, _, _ = model.context_encoder(user_id, hist_item_ids, num_weight_forget)
            if use_multi_interest and hasattr(model.context_encoder, "get_multi_interest_vectors"):
                multi_interests = model.context_encoder.get_multi_interest_vectors()
                if multi_interests is None:
                    multi_interests = common_prefer.unsqueeze(1)
            else:
                multi_interests = common_prefer.unsqueeze(1)

        target_emb = item_embeds[target_ids]  # [B, C, D]
        if use_multi_interest and hasattr(model.context_encoder, "score_items_with_interests"):
            scores, _ = model.context_encoder.score_items_with_interests(multi_interests, target_emb)
        else:
            scores_k = torch.einsum("bkd,bcd->bkc", multi_interests, target_emb)
            scores = scores_k.max(dim=1)[0]  # [B, C]

        pos_score = scores[:, 0:1]
        neg_scores = scores[:, 1:-1]
        ranks = (neg_scores > pos_score).long().sum(dim=1, keepdim=False)

        for rank in ranks:
            if rank < 1:
                current_metrics["ndcg_1"] += 1
                current_metrics["hit_1"] += 1
            if rank < 5:
                current_metrics["ndcg_5"] += 1 / torch.log2(rank + 2)
                current_metrics["hit_5"] += 1
            if rank < 10:
                current_metrics["ndcg_10"] += 1 / torch.log2(rank + 2)
                current_metrics["hit_10"] += 1
            if rank < 20:
                current_metrics["ndcg_20"] += 1 / torch.log2(rank + 2)
                current_metrics["hit_20"] += 1
            current_metrics["ap"] += 1.0 / (rank + 1)
        test_size += user_id.shape[0]

    for metric in list(current_metrics.keys()):
        current_metrics[metric] = float(current_metrics[metric] / test_size)
    return current_metrics


def main():
    print("=" * 20 + " BASELINE IOSC (UNIFIED CANDIDATE, K=1) " + "=" * 20)
    baseline_trainer = build_and_train(
        QuickTrainerBase, IOSC_encoder, orth_lambda=0.0, warmup_epochs=0, rec_model="IOSC"
    )
    baseline_metrics = evaluate_with_candidate_routing(
        baseline_trainer._model_, baseline_trainer._evaluator_1.test_batches, num_weight_forget=0.5, use_multi_interest=False
    )

    print("=" * 20 + " FUZZY IOSC (ATTN MEMBERSHIP + TARGET-AWARE) " + "=" * 20)
    fuzzy_trainer = build_and_train(
        QuickTrainerFuzzy,
        IOSC_encoder_Fuzzy,
        orth_lambda=0.01,
        warmup_epochs=1,
        rec_model="IOSC",
        corr_loss_weight=1.0,
    )
    fuzzy_metrics = evaluate_with_candidate_routing(
        fuzzy_trainer._model_, fuzzy_trainer._evaluator_1.test_batches, num_weight_forget=0.5, use_multi_interest=True
    )

    print("=" * 20 + " BERT4REC (UNIFIED CANDIDATE, K=1) " + "=" * 20)
    bert_trainer = build_and_train(
        QuickTrainerBase, BERT_encoder, orth_lambda=0.0, warmup_epochs=0, rec_model="BERT"
    )
    bert_metrics = evaluate_with_candidate_routing(
        bert_trainer._model_, bert_trainer._evaluator_1.test_batches, num_weight_forget=0.5, use_multi_interest=False
    )

    print("=" * 20 + " COMPARISON " + "=" * 20)
    print("unified_eval: candidate routing, baseline_K=1, fuzzy_K>1(max_k)")
    print("fuzzy_config: orth_lambda=0.01, warmup_epochs=1, corr_loss_weight=1.0, fuzzy_num_clusters=4, use_interest_specific_scorer=True, target_aware=max_k, modified_index=multi-interest reliability")
    keys = ["ndcg_10", "hit_10", "ap", "ndcg_5", "hit_5", "ndcg_20", "hit_20"]
    for key in keys:
        b = baseline_metrics[key]
        f = fuzzy_metrics[key]
        bert = bert_metrics[key]
        delta_fb = f - b
        delta_fbert = f - bert
        print(
            f"{key}: baseline={b:.6f}, bert4rec={bert:.6f}, fuzzy={f:.6f}, "
            f"fuzzy-baseline={delta_fb:+.6f}, fuzzy-bert={delta_fbert:+.6f}"
        )

    print("=" * 20 + " FUZZY ABLATION (corr_loss_weight=0) " + "=" * 20)
    fuzzy_no_corr_trainer = build_and_train(
        QuickTrainerFuzzy,
        IOSC_encoder_Fuzzy,
        orth_lambda=0.01,
        warmup_epochs=1,
        rec_model="IOSC",
        corr_loss_weight=0.0,
    )
    fuzzy_no_corr_metrics = evaluate_with_candidate_routing(
        fuzzy_no_corr_trainer._model_,
        fuzzy_no_corr_trainer._evaluator_1.test_batches,
        num_weight_forget=0.5,
        use_multi_interest=True,
    )
    for key in keys:
        f = fuzzy_metrics[key]
        f0 = fuzzy_no_corr_metrics[key]
        delta = f0 - f
        print(f"{key}: corr=1.0 {f:.6f}, corr=0.0 {f0:.6f}, delta={delta:+.6f}")


if __name__ == "__main__":
    main()
