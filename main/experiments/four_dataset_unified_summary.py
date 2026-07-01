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
from model.ComiRec import ComiRec_encoder
from model.IOSCencoder_fuzzy import IOSC_encoder_Fuzzy
import compare_iosc_vs_fuzzy_effective as cmp
from multi_dataset_benchmark import (
    evaluate_with_builtin_eval_ranking,
    prepare_cfg,
    steam_cfg,
    yelp_cfg,
    ele_cfg,
    maybe_attach_item_importance,
)


def seed_all(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eval_high_budget(cfg, model_name, max_train_batches=120):
    seed_all(123)
    cmp.patch_short_epoch_forget_schedule()

    if model_name == "ComiRec":
        cfg["rec_model"] = "COMIREC"
        trainer_cls = cmp.QuickTrainerBase
        trainer_module.IOSC_encoder = ComiRec_encoder
    elif model_name == "Fuzzy":
        cfg["rec_model"] = "IOSC"
        cfg = maybe_attach_item_importance(cfg)
        cmp.patch_iosc_device_behavior()
        trainer_cls = cmp.QuickTrainerFuzzy
        trainer_module.IOSC_encoder = IOSC_encoder_Fuzzy
    else:
        raise ValueError(model_name)

    data_model = SeqDataCollector(cfg)
    trainer = trainer_cls(
        cfg,
        data_model,
        save_dir="./datasets/" + cfg["dataset"] + "/seq/",
        max_train_batches=max_train_batches,
    )
    trainer.run_co()
    return evaluate_with_builtin_eval_ranking(trainer._model_, trainer._evaluator_1.test_batches)


def main():
    # Unified budget for all datasets/models.
    budget_epoch_num = 6
    budget_max_train_batches = 120

    dataset_cfgs = {
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
        "cd": prepare_cfg(
            {
                **steam_cfg,
                "dataset": "cd",
                "input_len": 10,
                "max_seq_len": 10,
            }
        ),
    }
    for ds in dataset_cfgs:
        dataset_cfgs[ds]["epoch_num"] = budget_epoch_num

    # "效果最好" fuzzy recipe used for cross-dataset summary:
    # (keeps strong accuracy from prior runs)
    fuzzy_cfg_patch = {
        "item_importance_method": "pagerank",
        "multi_interest_loss_weight": 1.0,
        "corr_loss_weight": 0.0,
        "orth_lambda": 0.0,
        "salience_importance_weight": 1.0,
        "salience_recency_weight": 0.0,
        "salience_confidence_weight": 0.0,
        "salience_fatigue_weight": 0.0,
        "fuzzy_num_clusters": 4,
        "interest_temperature": 1.0,
        "use_interest_specific_scorer": True,
    }

    results = {}
    for ds, cfg in dataset_cfgs.items():
        results[ds] = {}
        print(f"\n===== Running ComiRec on {ds} =====")
        results[ds]["ComiRec"] = train_eval_high_budget(
            copy.deepcopy(cfg), "ComiRec", max_train_batches=budget_max_train_batches
        )

        fuzzy_cfg = copy.deepcopy(cfg)
        fuzzy_cfg.update(fuzzy_cfg_patch)
        print(f"\n===== Running Fuzzy on {ds} =====")
        results[ds]["Fuzzy"] = train_eval_high_budget(
            fuzzy_cfg, "Fuzzy", max_train_batches=budget_max_train_batches
        )

    print("\n=== FOUR_DATASET_UNIFIED_TABLE ===")
    print(
        f"budget: epoch_num={budget_epoch_num}, max_train_batches={budget_max_train_batches} | "
        "eval=project_builtin_eval_ranking"
    )
    print("| Dataset | Model | NDCG@10 | Hit@10 | AP | NDCG@5 | Hit@5 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for ds in dataset_cfgs.keys():
        for m in ["ComiRec", "Fuzzy"]:
            met = results[ds][m]
            print(
                f"| {ds} | {m} | {met['ndcg_10']:.6f} | {met['hit_10']:.6f} | "
                f"{met['ap']:.6f} | {met['ndcg_5']:.6f} | {met['hit_5']:.6f} |"
            )

    print("\n=== DELTA_FUZZY_MINUS_COMIREC ===")
    for ds in dataset_cfgs.keys():
        delta = results[ds]["Fuzzy"]["ndcg_10"] - results[ds]["ComiRec"]["ndcg_10"]
        print(f"{ds}: delta_ndcg10={delta:+.6f}")


if __name__ == "__main__":
    main()
