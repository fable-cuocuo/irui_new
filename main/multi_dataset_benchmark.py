import copy
import random
import numpy as np
import torch

from data_utils.SeqDataGenerator import SeqDataCollector
import model_utils.Trainer as trainer_module
from model.IOSCencoder import IOSC_encoder
from model.IOSCencoder_fuzzy import IOSC_encoder_Fuzzy
from model.Bert4Rec import BERT_encoder

import compare_iosc_vs_fuzzy_effective as cmp
from steam_main import config as steam_cfg
from yelp_main import config as yelp_cfg
from ele_main import config as ele_cfg
from item_importance_utils import build_item_importance_scores


def seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cfg(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["train_type"] = "train"
    cfg["save_epochs"] = []
    cfg["epoch_num"] = 3
    cfg["train_batch_size"] = 512
    cfg["test_batch_size"] = 512
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "max_seq_len" not in cfg:
        cfg["max_seq_len"] = cfg["input_len"]

    # Fuzzy-specific params (ignored by non-fuzzy models)
    cfg["fuzzy_num_clusters"] = 4
    cfg["interest_temperature"] = 1.0
    cfg["use_interest_specific_scorer"] = True
    cfg["orth_lambda"] = 0.01
    cfg["orth_warmup_epochs"] = 1
    cfg["orth_warmup_max_scale"] = 1.0
    cfg["multi_interest_loss_weight"] = 0.3
    cfg["corr_loss_weight"] = 1.0
    cfg["reliability_threshold"] = 0.15
    cfg["salience_temperature"] = 1.0
    cfg["item_importance_method"] = "popularity"
    return cfg


_ITEM_IMPORTANCE_CACHE = {}


def maybe_attach_item_importance(cfg):
    if cfg.get("rec_model") != "IOSC":
        return cfg
    if "item_importance_scores" in cfg:
        return cfg
    method = cfg.get("item_importance_method", "popularity")
    dataset = cfg["dataset"]
    cache_key = (dataset, method)
    if cache_key not in _ITEM_IMPORTANCE_CACHE:
        seq_path = f"./datasets/{dataset}/seq/seq.dat"
        _ITEM_IMPORTANCE_CACHE[cache_key] = build_item_importance_scores(seq_path, method=method)
    cfg["item_importance_scores"] = _ITEM_IMPORTANCE_CACHE[cache_key]
    return cfg


def train_eval(cfg, model_name):
    seed_all(123)
    cmp.patch_short_epoch_forget_schedule()
    if model_name in ("IOSC", "Fuzzy"):
        cmp.patch_iosc_device_behavior()

    if model_name == "IOSC":
        cfg["rec_model"] = "IOSC"
        trainer_cls = cmp.QuickTrainerBase
        trainer_module.IOSC_encoder = IOSC_encoder
        encoder = IOSC_encoder
        use_multi_interest = False
    elif model_name == "Fuzzy":
        cfg["rec_model"] = "IOSC"
        cfg = maybe_attach_item_importance(cfg)
        trainer_cls = cmp.QuickTrainerFuzzy
        trainer_module.IOSC_encoder = IOSC_encoder_Fuzzy
        encoder = IOSC_encoder_Fuzzy
        use_multi_interest = True
    elif model_name == "BERT4Rec":
        cfg["rec_model"] = "BERT"
        trainer_cls = cmp.QuickTrainerBase
        encoder = BERT_encoder
        use_multi_interest = False
    else:
        raise ValueError(model_name)

    data_model = SeqDataCollector(cfg)
    trainer = trainer_cls(
        cfg,
        data_model,
        save_dir="./datasets/" + cfg["dataset"] + "/seq/",
        max_train_batches=60,
    )
    trainer.run_co()
    metrics = cmp.evaluate_with_candidate_routing(
        trainer._model_,
        trainer._evaluator_1.test_batches,
        num_weight_forget=0.5,
        use_multi_interest=use_multi_interest,
    )
    return metrics


def main():
    dataset_cfgs = {
        "steam": prepare_cfg(steam_cfg),
        "yelp": prepare_cfg(yelp_cfg),
        "electronics": prepare_cfg(ele_cfg),
        "ml1m": prepare_cfg(
            {
                **steam_cfg,
                "dataset": "ml1m",
                "input_len": 10,
                "max_seq_len": 10,
            }
        ),
    }
    models = ["BERT4Rec", "Fuzzy"]

    results = {}
    for ds_name, cfg in dataset_cfgs.items():
        results[ds_name] = {}
        for m in models:
            print(f"\n===== Running {m} on {ds_name} =====")
            results[ds_name][m] = train_eval(copy.deepcopy(cfg), m)

    print("\n=== FINAL_TABLE ===")
    print("| Dataset | Model | NDCG@10 | Hit@10 | AP | NDCG@5 | Hit@5 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for ds_name in dataset_cfgs.keys():
        for m in models:
            met = results[ds_name][m]
            print(
                f"| {ds_name} | {m} | {met['ndcg_10']:.6f} | {met['hit_10']:.6f} | "
                f"{met['ap']:.6f} | {met['ndcg_5']:.6f} | {met['hit_5']:.6f} |"
            )


if __name__ == "__main__":
    main()
