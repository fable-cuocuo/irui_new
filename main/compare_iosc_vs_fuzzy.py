import copy
import random
import numpy as np
import torch

from data_utils.SeqDataGenerator import SeqDataCollector
import model_utils.Trainer as trainer_module
from model_utils.Trainer import Trainer
from model.IOSCencoder import IOSC_encoder
from model.IOSCencoder_fuzzy import IOSC_encoder_Fuzzy
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
    return cfg


class QuickTrainer(Trainer):
    def __init__(self, config, data_model, save_dir, max_train_batches=80):
        self.max_train_batches = max_train_batches
        super().__init__(config, data_model, save_dir)

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
                loss, rec_loss_isc, rec_loss_osc, _, _ = self.train_one_batch(
                    user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_indices, epoch
                )
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


def run_single(name: str, encoder_cls):
    cfg = build_config()
    seed_all(123)

    # Avoid Trainer's hard-coded warmup length assumption when epoch_num < 20.
    trainer_module.Trainer.build_high_forget_rates = (
        lambda self: np.ones(self.config["epoch_num"]) * self.config["high_loss_drop"]
    )
    trainer_module.Trainer.build_low_forget_rates = (
        lambda self: np.ones(self.config["epoch_num"]) * self.config["low_loss_drop"]
    )
    trainer_module.Trainer.build_weight_forget_rates = (
        lambda self: np.ones(self.config["epoch_num"]) * self.config["weight_drop"]
    )
    patch_iosc_device_behavior()

    trainer_module.IOSC_encoder = encoder_cls
    data_model = SeqDataCollector(cfg)
    trainer = QuickTrainer(cfg, data_model, save_dir="./datasets/" + cfg["dataset"] + "/seq/")
    trainer.run_co()
    metrics = trainer._evaluator_1.best_metrics
    return {k: float(v) for k, v in metrics.items()}


def main():
    print("=" * 20 + " BASELINE IOSC " + "=" * 20)
    baseline_metrics = run_single("IOSC", IOSC_encoder)
    print("=" * 20 + " FUZZY IOSC " + "=" * 20)
    fuzzy_metrics = run_single("IOSC_Fuzzy", IOSC_encoder_Fuzzy)

    print("=" * 20 + " COMPARISON " + "=" * 20)
    keys = ["ndcg_10", "hit_10", "ap", "ndcg_5", "hit_5", "ndcg_20", "hit_20"]
    for key in keys:
        b = baseline_metrics[key]
        f = fuzzy_metrics[key]
        delta = f - b
        print(f"{key}: baseline={b:.6f}, fuzzy={f:.6f}, delta={delta:+.6f}")


if __name__ == "__main__":
    main()
