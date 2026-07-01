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
    yelp_cfg,
    maybe_attach_item_importance,
    steam_cfg,
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
    metrics = evaluate_with_builtin_eval_ranking(trainer._model_, trainer._evaluator_1.test_batches)
    return metrics


def main():
    dataset_cfgs = {
        "yelp": prepare_cfg(yelp_cfg),
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
        dataset_cfgs[ds]["epoch_num"] = 6

    # ComiRec baseline references under the same training budget.
    comirec_ref = {}
    for ds, cfg in dataset_cfgs.items():
        print(f"\n===== Running ComiRec baseline ({ds}) =====")
        comirec_ref[ds] = train_eval_high_budget(copy.deepcopy(cfg), "ComiRec", max_train_batches=120)

    # Innovation-preserving variants:
    # - keep orthogonal decoupling (orth_lambda > 0)
    # - keep multi-factor salience (importance + recency + confidence; optional fatigue)
    variants = {
        "inv_v1_light_orth_mf_sal": {
            "item_importance_method": "pagerank",
            "multi_interest_loss_weight": 1.0,
            "corr_loss_weight": 0.0,
            "orth_lambda": 0.005,
            "salience_importance_weight": 1.0,
            "salience_recency_weight": 0.3,
            "salience_confidence_weight": 0.2,
            "salience_fatigue_weight": 0.05,
            "recency_gamma": 0.5,
            "fatigue_window": 4,
        },
        "inv_v2_balanced_orth_mf_sal": {
            "item_importance_method": "pagerank",
            "multi_interest_loss_weight": 1.0,
            "corr_loss_weight": 0.0,
            "orth_lambda": 0.01,
            "salience_importance_weight": 1.0,
            "salience_recency_weight": 0.5,
            "salience_confidence_weight": 0.3,
            "salience_fatigue_weight": 0.10,
            "recency_gamma": 0.5,
            "fatigue_window": 4,
        },
        "inv_v3_strong_orth_mf_sal": {
            "item_importance_method": "pagerank",
            "multi_interest_loss_weight": 1.0,
            "corr_loss_weight": 0.0,
            "orth_lambda": 0.02,
            "salience_importance_weight": 1.0,
            "salience_recency_weight": 0.7,
            "salience_confidence_weight": 0.4,
            "salience_fatigue_weight": 0.15,
            "recency_gamma": 0.6,
            "fatigue_window": 4,
        },
        "inv_v4_with_corr": {
            "item_importance_method": "pagerank",
            "multi_interest_loss_weight": 1.0,
            "corr_loss_weight": 0.3,
            "orth_lambda": 0.01,
            "salience_importance_weight": 1.0,
            "salience_recency_weight": 0.5,
            "salience_confidence_weight": 0.3,
            "salience_fatigue_weight": 0.10,
            "recency_gamma": 0.5,
            "fatigue_window": 4,
        },
        "inv_v5_blend_importance": {
            "item_importance_method": "blend:popularity:0.5,pagerank:0.3,eigenvector:0.2",
            "multi_interest_loss_weight": 1.0,
            "corr_loss_weight": 0.0,
            "orth_lambda": 0.01,
            "salience_importance_weight": 1.0,
            "salience_recency_weight": 0.5,
            "salience_confidence_weight": 0.3,
            "salience_fatigue_weight": 0.10,
            "recency_gamma": 0.5,
            "fatigue_window": 4,
        },
    }

    results = {}
    for v_name, v_cfg in variants.items():
        results[v_name] = {}
        for ds, base_cfg in dataset_cfgs.items():
            run_cfg = copy.deepcopy(base_cfg)
            run_cfg.update(v_cfg)
            print(f"\n===== Running {v_name} on {ds} =====")
            results[v_name][ds] = train_eval_high_budget(run_cfg, "Fuzzy", max_train_batches=120)

    print("\n=== INNOVATION_PRESERVING_TABLE ===")
    print("| Variant | Dataset | NDCG@10 | Hit@10 | AP | NDCG@5 | Hit@5 | DeltaNDCG10_vs_ComiRec |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for v_name in variants.keys():
        for ds in dataset_cfgs.keys():
            met = results[v_name][ds]
            delta = met["ndcg_10"] - comirec_ref[ds]["ndcg_10"]
            print(
                f"| {v_name} | {ds} | {met['ndcg_10']:.6f} | {met['hit_10']:.6f} | {met['ap']:.6f} | "
                f"{met['ndcg_5']:.6f} | {met['hit_5']:.6f} | {delta:+.6f} |"
            )

    print("\n=== INNOVATION_PRESERVING_SUMMARY ===")
    print("| Variant | Avg NDCG@10 | Avg Delta vs ComiRec |")
    print("|---|---:|---:|")
    best_name = None
    best_avg_ndcg = -1.0
    best_avg_delta = None
    for v_name in variants.keys():
        avg_ndcg = float(np.mean([results[v_name][ds]["ndcg_10"] for ds in dataset_cfgs.keys()]))
        avg_delta = float(
            np.mean([results[v_name][ds]["ndcg_10"] - comirec_ref[ds]["ndcg_10"] for ds in dataset_cfgs.keys()])
        )
        print(f"| {v_name} | {avg_ndcg:.6f} | {avg_delta:+.6f} |")
        if avg_ndcg > best_avg_ndcg:
            best_avg_ndcg = avg_ndcg
            best_avg_delta = avg_delta
            best_name = v_name

    print("\n=== COMIREC_REFERENCE ===")
    for ds in dataset_cfgs.keys():
        ref = comirec_ref[ds]
        print(f"{ds}: ndcg_10={ref['ndcg_10']:.6f}, hit_10={ref['hit_10']:.6f}, ap={ref['ap']:.6f}")

    print(f"\nBEST_INNOVATION_VARIANT={best_name}")
    print(f"BEST_INNOVATION_AVG_NDCG10={best_avg_ndcg:.6f}")
    print(f"BEST_INNOVATION_AVG_DELTA_VS_COMIREC={best_avg_delta:+.6f}")


if __name__ == "__main__":
    main()
