import os
import random
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(CURRENT_DIR)
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)

from model_utils.Trainer import Trainer
from data_utils.SeqDataGenerator import SeqDataCollector
from data_utils.RankingEvaluator import print_dict

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = {
    # data settings
    "thread_num": 4,
    "dataset": "electronics",
    "eval_neg_num": 99,
    "train_neg_num": 30,
    "input_len": 10,

    # training settings
    "rec_model": "MIND",  # MIND / COMIREC / BERT / SASRec / IOSC / FPMC
    "train_type": "train",  # train / eval
    "save_epochs": [100, 200, 300, 400],
    "epoch_num": 500,
    "learning_rate": 0.001,
    "train_batch_size": 1024,
    "test_batch_size": 512,
    "drop_ratio": 0.1,

    # network settings
    "decay_factor": 0.9,
    "hidden_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 100,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "type_vocab_size": 1,
    "initializer_range": 0.1,
    "loss_type": "pairwise_sample",
    "weight_decay": 0.01,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max_seq_len": 10,
    "layer_size": 1,

    # MIND-specific
    "num_interest": 4,
    "mind_routing_iters": 3,
    "mind_add_pos": True,
    "mind_routing_eps": 1e-9,

    # legacy params (kept for compatibility with existing trainer)
    "threshold": 1,
    "noise_ratio": 0.05,
    "candidate_num": 3,
    "high_loss_drop": 0.15,
    "low_loss_drop": 0.15,
    "weight_drop": 0.5,
    "encoder_loss": False,
}

random.seed(123)


def main():
    data_model = SeqDataCollector(config)
    print_dict(config, "config")
    trainer = Trainer(config, data_model, save_dir="./datasets/" + config["dataset"] + "/seq/")
    trainer.run_co()


if __name__ == "__main__":
    main()
