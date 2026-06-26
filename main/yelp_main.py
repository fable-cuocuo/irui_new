import os
import random
import torch
from model_utils.Trainer import Trainer
from data_utils.SeqDataGenerator import SeqDataCollector
from data_utils.RankingEvaluator import print_dict

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




config = {
    # data settings
    'thread_num': 4,
    'dataset': 'yelp',
    'eval_neg_num': 99,
    'train_neg_num': 15,
    'input_len': 30,

    
    # training settings
    'rec_model': 'IOSC',# BERT SASRec FPMC
    'train_type': 'train',  # train / eval
    'save_epochs': [100,200,300,400],
    'epoch_num': 500,
    'learning_rate': 0.01,
    'train_batch_size': 1024,
    'test_batch_size': 256,
    'drop_ratio': 0.1,
    # net setting
    'decay_factor': 0.9,
    'hidden_size': 50,
    'num_hidden_layers': 1,
    'num_attention_heads': 2,
    'intermediate_size': 100,
    'hidden_act': "gelu",
    'hidden_dropout_prob': 0.1,
    'attention_probs_dropout_prob': 0.1,
    'type_vocab_size': 1,
    'initializer_range': 0.1,
    'loss_type': 'pairwise_sample',
    'weight_decay': 0.01,
    'device': torch.device('cuda'),
    'max_seq_len': 30,
    'layer_size': 1,

    # no used
    'threshold': 1,
    'noise_ratio': 0.05,  # 0.05 for steam and cd
    
    # xyz setting
    'candidate_num': 3,
    'candidate_num_2': 5,
    'high_loss_drop': 0.1,
    'low_loss_drop': 0.05,
    'weight_drop': 0.5,
    'encoder_loss': False

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
