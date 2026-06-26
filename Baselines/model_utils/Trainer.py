from data_utils.RankingEvaluator import RankingEvaluator
import torch
from torch import optim
import time
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.SeqRec import SeqRec
from model.FPMC import FPMC_encoder
from model.SASRec import SASRec_encoder
from model.Bert4Rec import BERT4Rec_encoder
from model.GRU4Rec import GRU4Rec_encoder
from model.Caser import Caser_encoder



class Trainer:
    def __init__(self, config, data_model, save_dir):
        # data_model:GraphDataCollector
        train_loader = data_model.generate_train_dataloader_unidirect()
        # valid_loader = data_model.generate_valid_dataloader_unidirect()
        test_loader = data_model.generate_test_dataloader_unidirect()

        self.item_num = data_model.numItem

        self.config = config
        self.save_dir = save_dir
        self.train_type = config['train_type']  # train/eval
        self.rec_model = config['rec_model']  # BERD
        # self.seq_model = seq_model
        self.train_loader = train_loader
        self._evaluator_1 = RankingEvaluator(test_loader)
        # self._evaluator_2 = RankingEvaluator(test_loader)
        self.item_dist = np.array(data_model.item_dist)
        self.user_dist = np.array(data_model.user_dist)
        self.train_size = len(train_loader.dataset)
        self.model_save_dir = './datasets/' + self.config['dataset'] + '/model/'
        self.model_save_path = self.model_save_dir + self.rec_model + str(self.config['sample_loss_weight']) + '-'
        self.save_epochs = self.config['save_epochs']

        seq_model = self.getSeqEncoder()
        rec_model = SeqRec(config, seq_model)

        if self.train_type == 'train':
            if rec_model is not None:
                self._model_ = rec_model
                self._device = config['device']
                self._model_.double().to(self._device)
                self._optimizer_ = _get_optimizer(
                    self._model_, learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
                self.scheduler = ReduceLROnPlateau(self._optimizer_, 'max', patience=10,
                                                   factor=config['decay_factor'])

        elif self.train_type == 'eval':
            self._device = config['device']
            self._model_ = rec_model

    def run_co(self):
        if self.train_type == 'train':
            print('=' * 60, '\n', 'Start Training', '\n', '=' * 60, sep='')
            keep_train = True
            for epoch in range(self.config['epoch_num']):
                start = time.time()
                loss_iter = 0
                for i, batch in enumerate(self.train_loader):
                    user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_indices = batch
                    loss = self.train_one_batch(user_id, hist_item_ids, masks, pos_target, train_candidates,
                                                neg_targets, sample_indices)
                    loss_iter += loss.item()
                print(f'################## epoch {epoch} ###########################')
                print(
                    f"loss: {round(loss_iter / len(self.train_loader), 4)}, len_train_loader:{len(self.train_loader)}")
                keep_train = self.evaluate(epoch)  # 验证
                print('#########################################################')
                if epoch in self.save_epochs:
                    self.save_model(epoch)
                if not keep_train:
                    break

        elif self.train_type == 'eval':
            for epoch in self.save_epochs:
                self._model_ = self.load_model(epoch)
                self._model_.double().to(self._device)
                self._evaluator_1.evaluate(model=self._model_, train_iter=0)

    def evaluate(self, iter):
        self._model_.eval()
        keep_train, ndcg10 = self._evaluator_1.evaluate(model=self._model_, train_iter=iter)
        self.scheduler.step(ndcg10)
        return keep_train

    def train_one_batch(self, user_id, hist_item_ids, masks, pos_target, train_candidates, neg_targets, sample_idices):
        self._model_.train()
        self._optimizer_.zero_grad()

        # [bs], [bs,1]
        loss = self._model_(user_id, hist_item_ids, masks, pos_target, neg_targets, sample_idices)
        # [bs]
        loss = loss.sum()
        loss.backward()
        self._optimizer_.step()
        return loss

    def select_unvalid_loss(self, loss, modified_index, list_len):
        ind_loss_sorted = np.argsort(loss.cpu().data)
        length = int(list_len * 0.05)
        changed_loss_ind = set(ind_loss_sorted[-length:].tolist())
        changed_loss_ind = list(changed_loss_ind)
        target_changed_bool = np.zeros(list_len)
        target_changed_bool[changed_loss_ind] = -2
        target_changed_bool = torch.tensor(target_changed_bool)

        if torch.cuda.is_available():
            target_changed_bool = target_changed_bool.to(torch.device('cuda'))

        changed_choice = torch.where(target_changed_bool == -2, target_changed_bool, modified_index)
        return changed_choice

    def save_model(self, epoch_num):
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        save_path = self.model_save_path + str(epoch_num) + '-model.pkl'
        torch.save(self._model_, save_path)
        print(f'model saved at {save_path}')

    def getSeqEncoder(self):
        if self.config['rec_model'] == 'SASRec':
            return SASRec_encoder(self.config)
        elif self.config['rec_model'] == 'FPMC':
            return FPMC_encoder(self.config)
        elif self.config['rec_model'] == 'GRU4Rec':
            return GRU4Rec_encoder(self.config)
        elif self.config['rec_model'] == 'BERT4Rec':
            return BERT4Rec_encoder(self.config)
        elif self.config['rec_model'] == 'Caser':
            return Caser_encoder(self.config)

    def load_model(self, epoch_num):
        load_path = self.model_save_path + str(epoch_num) + '-model.pkl'
        print(f'loading model from {load_path}')
        return torch.load(load_path)


def _get_optimizer(model, learning_rate, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
