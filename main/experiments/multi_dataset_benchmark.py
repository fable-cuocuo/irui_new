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
from model.IOSCencoder import IOSC_encoder
from model.IOSCencoder_fuzzy import IOSC_encoder_Fuzzy
from model.Bert4Rec import BERT_encoder
from model.ComiRec import ComiRec_encoder
from model.MIND import MIND_encoder

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
    cfg["item_importance_method"] = "pagerank"
    cfg["salience_importance_weight"] = 1.0
    cfg["salience_recency_weight"] = 0.5
    cfg["salience_confidence_weight"] = 0.3
    cfg["salience_fatigue_weight"] = 0.25
    cfg["recency_gamma"] = 0.5
    cfg["fatigue_window"] = 4
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


@torch.no_grad()
def evaluate_with_builtin_eval_ranking(model, test_batches):
    model.eval()
    metrics = {
        "ndcg_1": 0.0,
        "ndcg_5": 0.0,
        "ndcg_10": 0.0,
        "ndcg_20": 0.0,
        "hit_1": 0.0,
        "hit_5": 0.0,
        "hit_10": 0.0,
        "hit_20": 0.0,
        "ap": 0.0,
    }
    test_size = 0
    for test_batch in test_batches:
        batch_metrics = model.eval_ranking(test_batch)
        for k, v in batch_metrics.items():
            metrics[k] += float(v)
        test_size += int(test_batch[0].shape[0])
    for k in list(metrics.keys()):
        metrics[k] = metrics[k] / max(test_size, 1)
    return metrics


def train_eval(cfg, model_name, eval_mode="routing"):
    seed_all(123)
    cmp.patch_short_epoch_forget_schedule()
    if model_name in ("IOSC", "Fuzzy"):
        cmp.patch_iosc_device_behavior()

    if model_name == "IOSC":
        cfg["rec_model"] = "IOSC"
        trainer_cls = cmp.QuickTrainerBase
        trainer_module.IOSC_encoder = IOSC_encoder
        use_multi_interest = False
    elif model_name == "Fuzzy":
        cfg["rec_model"] = "IOSC"
        cfg = maybe_attach_item_importance(cfg)
        trainer_cls = cmp.QuickTrainerFuzzy
        trainer_module.IOSC_encoder = IOSC_encoder_Fuzzy
        use_multi_interest = True
    elif model_name == "BERT4Rec":
        cfg["rec_model"] = "BERT"
        trainer_cls = cmp.QuickTrainerBase
        use_multi_interest = False
    elif model_name == "ComiRec":
        cfg["rec_model"] = "COMIREC"
        trainer_cls = cmp.QuickTrainerBase
        trainer_module.IOSC_encoder = ComiRec_encoder
        use_multi_interest = True
    elif model_name == "MIND":
        cfg["rec_model"] = "MIND"
        trainer_cls = cmp.QuickTrainerBase
        trainer_module.IOSC_encoder = MIND_encoder
        use_multi_interest = True
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
    if eval_mode == "builtin":
        metrics = evaluate_with_builtin_eval_ranking(
            trainer._model_,
            trainer._evaluator_1.test_batches,
        )
    else:
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
    models = ["BERT4Rec", "ComiRec", "MIND", "Fuzzy"]

    results = {}
    for ds_name, cfg in dataset_cfgs.items():
        results[ds_name] = {}
        for m in models:
            print(f"\n===== Running {m} on {ds_name} =====")
            results[ds_name][m] = train_eval(copy.deepcopy(cfg), m, eval_mode="builtin")

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
