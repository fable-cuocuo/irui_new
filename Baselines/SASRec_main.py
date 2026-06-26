import os
import random
import torch
from model_utils.Trainer import Trainer
from data_utils.SeqDataGenerator import SeqDataCollector
from data_utils.RankingEvaluator import print_dict

torch.multiprocessing.set_sharing_strategy('file_system')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    config = {
        # data settings
        'thread_num': 4,
        'dataset': 'electronics',
        'eval_neg_num': 20,
        'train_neg_num': 5,
        'input_len': 5,

        'candidate_num': 4,
        'mask_prob': 0.5,
        'rec_model': 'GRU4Rec',
        'train_type': 'train',  # train / eval
        'save_epochs': [8],
        'epoch_num': 500,
        'learning_rate': 0.01,
        'train_batch_size': 256,
        'test_batch_size': 256,
        'drop_ratio': 0.1,

        'hidden_size': 50,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 100,
        'hidden_act': "gelu",
        'hidden_dropout_prob': 0.1,

        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'max_seq_len': 200,
        'layer_size': 1,
        'add_user': True,
        # no used
        'threshold': 1,
        'noise_ratio': 0.05,  # 0.05 for steam and cd
        # training settings

        # BPR settings
        'factor_num': 32,
        # graph settings
        'n_layers': 2,
        'next_hop_num': 1,
        # prob settings
        'entropy_threshold': 1.0,
        'sample_num': 4,
        'sample_loss_weight': 0.01,
        # BERT settings
        'decay_factor': 0.9,

        'attention_probs_dropout_prob': 0.1,
        'type_vocab_size': 1,
        'initializer_range': 0.1,
        'loss_type': 'pairwise_sample',
        'weight_decay': 0.01,
        # caser setting
        'n_h': 16,
        'n_v': 4,
        # GRU4Rec config
        'gru_layer_num': 2,
        'candidate_size': 2,

    }

random.seed(123)


def main():
    # './datasets/electronics/seq/', './datasets/sports/seq/', './datasets/ml2k/seq/'
    # 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
    # 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    data_model = SeqDataCollector(config)
    print_dict(config, 'config')
    trainer = Trainer(config, data_model, save_dir='./datasets/' + config['dataset'] + '/seq/')
    trainer.run_co()


if __name__ == '__main__':
    main()
